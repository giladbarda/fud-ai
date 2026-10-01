import Foundation
import UIKit
#if canImport(FoundationModels)
import FoundationModels
#endif
/// Routes a multi-turn chat (system context + user/assistant message history + new user message)
/// to the currently-selected LLM provider, with **tool calling** so the model can fetch any
/// historical slice of the user's data on demand instead of receiving a fixed-size dump in
/// the system prompt. Tool definitions live next to each provider's HTTP layer (Gemini /
/// Anthropic / OpenAI-compatible) and the executor lives in CoachTools.
struct ChatService {
    enum ChatError: LocalizedError {
        case noAPIKey
        case networkError(Error)
        case apiError(String)
        case invalidResponse
        var errorDescription: String? {
            switch self {
            case .noAPIKey:
                return "No API key configured. Add your key in Settings → AI Provider."
            case .networkError(let err):
                return "Network error: \(err.localizedDescription)"
            case .apiError(let msg):
                return "API error: \(msg)"
            case .invalidResponse:
                return "Could not understand the AI response. Please try again."
            }
        }
    }
    /// Hard cap on the number of tool-call rounds per user message. Generous —
    /// most real questions resolve in 1–2 calls (e.g. summary → range fetch →
    /// answer). Without this cap a misbehaving model could loop forever on
    /// recursive calls.
    private static let maxToolRounds = 6
    // MARK: - Public entry point
    static func sendMessage(
        history: [ChatMessage],
        newUserMessage: String,
        imageData: Data? = nil,
        profile: UserProfile,
        weights: [WeightEntry],
        bodyFats: [BodyFatEntry],
        measurements: [BodyMeasurement] = [],
        foods: [FoodEntry],
        fastingSessions: [FastingSession] = [],
        heightMetric: Bool,
        weightMetric: Bool,
        workoutSessions: [StrengthWorkoutSession] = [],
        workoutPlans: [StrengthWorkoutDayPlan] = [],
        workoutPreferences: StrengthWorkoutPreferences? = nil,
        workoutAccessEnabled: Bool = false
    ) async throws -> String {
        if AIModeSettings.isHosted {
            return try await hostedCoachMessage(
                history: history,
                newUserMessage: newUserMessage,
                imageData: imageData,
                profile: profile,
                weights: weights,
                bodyFats: bodyFats,
                measurements: measurements,
                foods: foods,
                fastingSessions: fastingSessions,
                heightMetric: heightMetric,
                weightMetric: weightMetric,
                workoutSessions: workoutSessions,
                workoutPlans: workoutPlans,
                workoutPreferences: workoutPreferences,
                workoutAccessEnabled: workoutAccessEnabled
            )
        }
        let systemPrompt = buildSystemPrompt(
            profile: profile,
            weights: weights,
            bodyFats: bodyFats,
            measurements: measurements,
            foods: foods,
            fastingSessions: fastingSessions,
            heightMetric: heightMetric,
            weightMetric: weightMetric,
            workoutSessions: workoutSessions,
            workoutPlans: workoutPlans,
            workoutAccessEnabled: workoutAccessEnabled
        )
        let tools = CoachTools(
            weights: weights,
            bodyFats: bodyFats,
            foods: foods,
            fastingSessions: fastingSessions,
            workoutSessions: workoutSessions,
            workoutPlans: workoutPlans,
            workoutPreferences: workoutPreferences,
            workoutPlanWeightUnit: weightMetric ? .kg : .lbs,
            workoutAccessEnabled: workoutAccessEnabled
        )
        let config = AIProviderSettings.currentConfig(requiresVision: imageData != nil)
        func request(
            provider: AIProvider,
            model: String,
            baseURL: String,
            apiKey: String?
        ) async throws -> String {
            guard !provider.requiresAPIKey || apiKey != nil else {
                throw ChatError.noAPIKey
            }
            switch provider.apiFormat {
            case .onDevice:
                return try await callOnDevice(
                    systemPrompt: systemPrompt,
                    history: history,
                    newUserMessage: newUserMessage,
                    imageData: imageData
                )
            case .liteRTLocal:
                let localHistory = history.suffix(8).map { message in
                    Gemma4LocalModelManager.ChatTurn(
                        role: message.role == .user ? .user : .assistant,
                        text: message.content
                    )
                }
                let localInstructions = """
                \(systemPrompt)
                ON-DEVICE MODE
                You cannot call data tools in this mode. Answer only from the profile, forecast, and data summary above. If the question requires un
