// applefm-shim: the Swift transport half of the Apple Foundation Models backend.
//
// One persistent subprocess speaking JSON-lines over stdin/stdout, spawned from
// applefm.py (see its module docstring for the FROZEN wire protocol -- this file
// implements it exactly). Requests arrive one JSON object per line, each carries
// an `id` that every response for it echoes back, and the process NEVER exits on
// a per-request error: it catches and emits an {"type":"error"} object instead,
// so a single bad request can't take down a long-lived shim the Python side is
// mid-conversation with.
//
// Ops: "availability" (SystemLanguageModel readiness), "respond" (a fresh
// LanguageModelSession per call -- callers never want history -- optionally with
// guided generation against a JSON-Schema subset, optionally streamed), and
// "embed" (NLContextualEmbedding, mean-pooled + L2-normalized). The chat
// ORCHESTRATION for the fixed 4,096-token window lives in Python (chat_apple.py);
// this file is only transport plus the two model bridges.
//
// Built by `make apple-shim` with `swiftc -O -parse-as-library` -- hence @main
// rather than top-level code. Command Line Tools alone suffice: both
// FoundationModels.framework and NaturalLanguage.framework ship in the CLT SDK.

import Foundation
import FoundationModels
import NaturalLanguage

// ---------------------------------------------------------------------------
// Output: one JSON object per line, flushed, every write serialized.
// ---------------------------------------------------------------------------
// A streamed `respond` emits several lines (chunks then done) and the main loop
// is serial, but the lock is still load-bearing as a contract guarantee: it is
// the single choke point that makes "exactly one well-formed JSON line at a
// time, never interleaved" true no matter how the call sites evolve. FileHandle
// writes go straight to the fd (no stdio buffering), so there is nothing to
// flush -- the line is on the wire when write() returns.
let outputLock = NSLock()

func emit(_ object: [String: Any]) {
    outputLock.lock()
    defer { outputLock.unlock() }
    guard let data = try? JSONSerialization.data(withJSONObject: object, options: []) else {
        // A response we can't even serialize (e.g. a non-finite float that
        // slipped through) must not crash the loop; drop it silently rather
        // than emit a half-written line the Python framer would choke on.
        return
    }
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data([0x0A]))  // newline terminates the frame
}

// A shim-originated error carrying the wire `kind` directly. FoundationModels'
// own GenerationError is mapped separately (see `emitError`).
struct ShimError: Error {
    let kind: String
    let message: String
    static func unavailable(_ m: String) -> ShimError { ShimError(kind: "unavailable", message: m) }
    static func badRequest(_ m: String) -> ShimError { ShimError(kind: "bad_request", message: m) }
}

func emitError(id: Any, _ error: Error) {
    let kind: String
    let message: String
    if let gen = error as? LanguageModelSession.GenerationError {
        // Three distinctions the Python side acts on. Two are fatal-ish:
        // exceededContextWindowSize is recoverable by retrying with less
        // (context_overflow), assetsUnavailable means the model is down
        // (unavailable, fatal). Every other GenerationError is a per-request,
        // content-driven rejection (unsupported language/locale, guardrail,
        // decoding failure, ...): recoverable by SKIPPING this input, so it maps
        // to generation_error rather than bad_request. bad_request stays
        // reserved for our own protocol/framing bugs (the NSError branch below
        // and ShimError.badRequest), which must remain fatal.
        switch gen {
        case .exceededContextWindowSize: kind = "context_overflow"
        case .assetsUnavailable: kind = "unavailable"
        default: kind = "generation_error"
        }
        message = gen.errorDescription ?? String(describing: gen)
    } else if let shim = error as? ShimError {
        kind = shim.kind
        message = shim.message
    } else {
        kind = "bad_request"
        message = (error as NSError).localizedDescription
    }
    emit(["id": id, "type": "error", "message": message, "kind": kind])
}

// The one model instance every respond call uses, configured with permissive
// content-transformation guardrails. This app transforms the user's OWN
// on-device messages (labeling, summarizing, term curation, answering questions
// about them) -- the exact use case Apple documents permissiveContentTransformations
// for -- and the default guardrail rejects a large fraction of real personal
// content (self-harm/violence/sexual message text) with .guardrailViolation,
// which would otherwise skip that content everywhere. Guardrails do not affect
// availability, so handleAvailability reads it off this same instance too. A
// file-scope `let` initializes lazily and thread-safely on first use, and is
// safe to construct even when Apple Intelligence is unavailable.
let generationModel = SystemLanguageModel(guardrails: .permissiveContentTransformations)

// ---------------------------------------------------------------------------
// availability
// ---------------------------------------------------------------------------
func handleAvailability(id: Any) {
    switch generationModel.availability {
    case .available:
        emit(["id": id, "type": "availability", "available": true, "reason": "available"])
    case .unavailable(let reason):
        // Human-readable strings the Python side surfaces verbatim in the
        // actionable error it raises when apple is selected but not usable.
        let text: String
        switch reason {
        case .deviceNotEligible:
            text = "device not eligible for Apple Intelligence"
        case .appleIntelligenceNotEnabled:
            text = "Apple Intelligence is not enabled in System Settings"
        case .modelNotReady:
            text = "model not ready (assets still downloading, or try again shortly)"
        @unknown default:
            text = "Apple Intelligence is unavailable"
        }
        emit(["id": id, "type": "availability", "available": false, "reason": text])
    @unknown default:
        emit(["id": id, "type": "availability", "available": false,
              "reason": "Apple Intelligence availability is unknown"])
    }
}

// ---------------------------------------------------------------------------
// JSON-Schema subset -> DynamicGenerationSchema
// ---------------------------------------------------------------------------
// Covers exactly what the app emits (config'd in tool_schemas.py): an object of
// properties (+ optional `required`), the scalar leaves string/integer/number/
// boolean, an enum of strings, and an array whose items are a scalar leaf.
// `description` is passed through; `default` is ignored. A `type` that is a JSON
// array (a union like ["string","null"]) or otherwise unexpected is narrowed to
// a single supported type rather than failing -- the Python side pre-simplifies,
// so this is only a safety net. Every sub-schema is inlined (no `referenceTo`),
// so GenerationSchema needs no dependencies.

let supportedTypes: Set<String> = ["object", "array", "string", "integer", "number", "boolean"]

// Resolve a node's effective type: honor an explicit scalar `type`, pick the
// first supported member of a union `type`, else infer structurally from the
// presence of `properties`/`items`, else fall back to string.
func resolveType(_ node: [String: Any]) -> String {
    if let s = node["type"] as? String, supportedTypes.contains(s) { return s }
    if let union = node["type"] as? [Any] {
        for entry in union where (entry as? String).map(supportedTypes.contains) == true {
            return entry as! String
        }
    }
    if node["properties"] != nil { return "object" }
    if node["items"] != nil { return "array" }
    return "string"
}

func buildSchema(_ node: [String: Any], name: String) -> DynamicGenerationSchema {
    let description = node["description"] as? String

    // enum-of-strings takes precedence over `type`: constrained decoding to a
    // fixed choice set (e.g. the route action) is the whole point.
    if let rawEnum = node["enum"] as? [Any] {
        let choices = rawEnum.compactMap { $0 as? String }
        if !choices.isEmpty {
            return DynamicGenerationSchema(name: name, description: description, anyOf: choices)
        }
    }

    switch resolveType(node) {
    case "object":
        let properties = node["properties"] as? [String: Any] ?? [:]
        let required = Set((node["required"] as? [Any])?.compactMap { $0 as? String } ?? [])
        // Deterministic property order (stable schema across identical inputs).
        let props = properties.keys.sorted().compactMap { key -> DynamicGenerationSchema.Property? in
            guard let child = properties[key] as? [String: Any] else { return nil }
            return DynamicGenerationSchema.Property(
                name: key,
                description: child["description"] as? String,
                schema: buildSchema(child, name: "\(name)_\(key)"),
                isOptional: !required.contains(key))
        }
        return DynamicGenerationSchema(name: name, description: description, properties: props)
    case "array":
        let items = node["items"] as? [String: Any] ?? ["type": "string"]
        return DynamicGenerationSchema(arrayOf: buildSchema(items, name: "\(name)_item"))
    case "integer":
        return DynamicGenerationSchema(type: Int.self)
    case "number":
        return DynamicGenerationSchema(type: Double.self)
    case "boolean":
        return DynamicGenerationSchema(type: Bool.self)
    default:  // "string" and the safety-net fallthrough
        return DynamicGenerationSchema(type: String.self)
    }
}

// ---------------------------------------------------------------------------
// respond
// ---------------------------------------------------------------------------
func handleRespond(id: Any, request: [String: Any]) async {
    let system = request["system"] as? String ?? ""
    let prompt = request["prompt"] as? String ?? ""
    let stream = request["stream"] as? Bool ?? false
    // Fresh session every call: the Python orchestrator carries all state
    // between turns itself, so history here would only burn the 4,096-token
    // window. `instructions` is the system message. `generationModel` carries
    // the permissive content-transformation guardrails (see its definition).
    let session = LanguageModelSession(model: generationModel, instructions: system)

    do {
        if let schemaNode = request["schema"] as? [String: Any] {
            // Guided generation: constrained decoding produces output that IS
            // valid JSON, serialized back out via GeneratedContent.jsonString.
            let root = buildSchema(schemaNode, name: "Root")
            let schema = try GenerationSchema(root: root, dependencies: [])
            let content = try await session.respond(to: prompt, schema: schema).content.jsonString
            if stream {
                emit(["id": id, "type": "chunk", "content": content])
                emit(["id": id, "type": "done"])
            } else {
                emit(["id": id, "type": "done", "content": content])
            }
        } else if stream {
            // streamResponse yields CUMULATIVE snapshots; the wire wants deltas,
            // so emit only the suffix grown since the previous snapshot.
            var previous = ""
            for try await snapshot in session.streamResponse(to: prompt) {
                let current = snapshot.content
                let delta = current.hasPrefix(previous)
                    ? String(current.dropFirst(previous.count))
                    : current
                if !delta.isEmpty {
                    emit(["id": id, "type": "chunk", "content": delta])
                }
                previous = current
            }
            emit(["id": id, "type": "done"])
        } else {
            let content = try await session.respond(to: prompt).content
            emit(["id": id, "type": "done", "content": content])
        }
    } catch {
        emitError(id: id, error)
    }
}

// ---------------------------------------------------------------------------
// embed
// ---------------------------------------------------------------------------
// One NLContextualEmbedding instance, lazily loaded and reused. Assets download
// over-the-air on first use; if they can't be obtained we surface a distinct
// "unavailable" error so the Python side can tell the user to check the network
// rather than reporting a generic failure.
var sharedEmbedder: NLContextualEmbedding?

func ensureEmbedder() async throws -> NLContextualEmbedding {
    if let existing = sharedEmbedder { return existing }
    guard let embedder = NLContextualEmbedding(script: .latin) else {
        throw ShimError.unavailable("no NLContextualEmbedding model for latin script on this OS")
    }
    if !embedder.hasAvailableAssets {
        let result = try await embedder.requestAssets()
        guard result == .available else {
            throw ShimError.unavailable("embedding assets unavailable (download failed -- check network)")
        }
    }
    try embedder.load()
    sharedEmbedder = embedder
    return embedder
}

// Mean-pool one segment's per-token vectors into a single dim-length vector.
// Returns nil for an empty/untokenizable segment so it contributes nothing.
func meanPoolTokens(_ text: String, embedder: NLContextualEmbedding, dim: Int) throws -> [Double]? {
    guard !text.isEmpty else { return nil }
    let result = try embedder.embeddingResult(for: text, language: .english)
    var sum = [Double](repeating: 0, count: dim)
    var count = 0
    result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, _ in
        if vector.count == dim {
            for i in 0..<dim { sum[i] += vector[i] }
            count += 1
        }
        return true  // true = keep enumerating (visit every token)
    }
    guard count > 0 else { return nil }
    for i in 0..<dim { sum[i] /= Double(count) }
    return sum
}

func embedOne(_ text: String, embedder: NLContextualEmbedding, dim: Int) throws -> [Double] {
    // The model truncates inputs past `maximumSequenceLength` TOKENS, so long
    // text would silently lose its tail. Segment first, mean-pool each segment,
    // then mean-pool the segment vectors. Segmenting by that many CHARACTERS is
    // a safe over-approximation: a subword token spans >= 1 character, so a
    // segment of N characters is at most N tokens -- never truncated.
    let maxLen = max(1, Int(embedder.maximumSequenceLength))
    let chars = Array(text)
    var segmentVectors: [[Double]] = []
    if chars.isEmpty {
        segmentVectors = []
    } else {
        var i = 0
        while i < chars.count {
            let end = min(i + maxLen, chars.count)
            if let pooled = try meanPoolTokens(String(chars[i..<end]), embedder: embedder, dim: dim) {
                segmentVectors.append(pooled)
            }
            i = end
        }
    }

    var pooled = [Double](repeating: 0, count: dim)
    guard !segmentVectors.isEmpty else { return pooled }  // all-zero: nothing to normalize
    for vector in segmentVectors {
        for i in 0..<dim { pooled[i] += vector[i] }
    }
    for i in 0..<dim { pooled[i] /= Double(segmentVectors.count) }

    // L2-normalize so downstream cosine distance is a dot product.
    let norm = sqrt(pooled.reduce(0) { $0 + $1 * $1 })
    guard norm > 0 else { return pooled }
    return pooled.map { $0 / norm }
}

func handleEmbed(id: Any, request: [String: Any]) async {
    guard let rawTexts = request["texts"] as? [Any] else {
        emitError(id: id, ShimError.badRequest("embed request missing `texts` array"))
        return
    }
    do {
        let embedder = try await ensureEmbedder()
        // The model's ACTUAL dimension -- never hardcoded. Python auto-detects
        // it from the first vector and dimensions chunk_vectors accordingly.
        let dim = Int(embedder.dimension)
        var vectors: [[Double]] = []
        vectors.reserveCapacity(rawTexts.count)
        for raw in rawTexts {
            vectors.append(try embedOne((raw as? String) ?? "", embedder: embedder, dim: dim))
        }
        emit(["id": id, "type": "embed", "vectors": vectors])
    } catch {
        emitError(id: id, error)
    }
}

// ---------------------------------------------------------------------------
// Dispatch + serial main loop
// ---------------------------------------------------------------------------
func handleLine(_ line: String) async {
    guard let data = line.data(using: .utf8),
          let parsed = try? JSONSerialization.jsonObject(with: data),
          let request = parsed as? [String: Any] else {
        // No id to correlate -- send it back as null so the framer stays happy.
        emit(["id": NSNull(), "type": "error",
              "message": "request line is not a JSON object", "kind": "bad_request"])
        return
    }
    let id = request["id"] ?? NSNull()
    guard let op = request["op"] as? String else {
        emit(["id": id, "type": "error", "message": "request missing `op`", "kind": "bad_request"])
        return
    }
    switch op {
    case "availability": handleAvailability(id: id)
    case "respond": await handleRespond(id: id, request: request)
    case "embed": await handleEmbed(id: id, request: request)
    default:
        emit(["id": id, "type": "error", "message": "unknown op: \(op)", "kind": "bad_request"])
    }
}

@main
struct ApplefmShim {
    // A single serial async loop: read a line (blocking), process it to
    // completion, repeat. Serial processing is a deliberate simplification --
    // FoundationModels serializes on-device inference anyway, and it removes
    // every response-ordering hazard without needing a task pool. Loop ends when
    // stdin closes (readLine returns nil), at which point the process exits 0.
    static func main() async {
        while let line = readLine(strippingNewline: true) {
            if line.isEmpty { continue }
            await handleLine(line)
        }
    }
}
