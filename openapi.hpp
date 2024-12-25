#pragma once
#include <string>
#include <vector>
#include <optional>

// If the generator can't map a type, we use UNKNOWN_TYPE to force a compile error.
#define UNKNOWN_TYPE static_assert(false, "Unknown schema type encountered!");

// Forward declarations:
struct AddUploadPartRequest;
struct AdminApiKey;
struct ApiKeyList;
struct AssistantObject;
struct AssistantsApiResponseFormatOption;
struct AssistantStreamEvent;
struct AssistantToolsCode;
struct AssistantToolsFileSearch;
struct AssistantToolsFileSearchTypeOnly;
struct AssistantToolsFunction;
struct FunctionObject;
struct FunctionParameters;
struct AssistantsApiToolChoiceOption;
struct AssistantsNamedToolChoice;
struct AudioResponseFormat;
struct AuditLog;
struct AuditLogEventType;
struct AuditLogActor;
struct AuditLogActorSession;
struct AuditLogActorUser;
struct AuditLogActorApiKey;
struct AuditLogActorServiceAccount;
struct AutoChunkingStrategyRequestParam;
struct Batch;
struct BatchRequestInput;
struct BatchRequestOutput;
struct CancelUploadRequest;
struct ChatCompletionFunctionCallOption;
struct ChatCompletionFunctions;
struct ChatCompletionMessageToolCall;
struct ChatCompletionMessageToolCallChunk;
struct ChatCompletionMessageToolCalls;
struct ChatCompletionModalities;
struct ChatCompletionNamedToolChoice;
struct ChatCompletionRequestAssistantMessage;
struct ChatCompletionRequestAssistantMessageContentPart;
struct ChatCompletionRequestDeveloperMessage;
struct ChatCompletionRequestFunctionMessage;
struct ChatCompletionRequestMessage;
struct ChatCompletionRequestMessageContentPartAudio;
struct ChatCompletionRequestMessageContentPartImage;
struct ChatCompletionRequestMessageContentPartRefusal;
struct ChatCompletionRequestMessageContentPartText;
struct ChatCompletionRequestSystemMessage;
struct ChatCompletionRequestSystemMessageContentPart;
struct ChatCompletionRequestToolMessage;
struct ChatCompletionRequestToolMessageContentPart;
struct ChatCompletionRequestUserMessage;
struct ChatCompletionRequestUserMessageContentPart;
struct ChatCompletionResponseMessage;
struct ChatCompletionRole;
struct ChatCompletionStreamOptions;
struct ChatCompletionStreamResponseDelta;
struct ChatCompletionTokenLogprob;
struct ChatCompletionTool;
struct ChatCompletionToolChoiceOption;
struct ChunkingStrategyRequestParam;
struct CompleteUploadRequest;
struct CompletionUsage;
struct CostsResult;
struct CreateAssistantRequest;
struct CreateChatCompletionFunctionResponse;
struct CreateChatCompletionImageResponse;
struct CreateChatCompletionRequest;
struct ParallelToolCalls;
struct CreateChatCompletionResponse;
struct CreateChatCompletionStreamResponse;
struct CreateCompletionRequest;
struct CreateCompletionResponse;
struct CreateEmbeddingRequest;
struct CreateEmbeddingResponse;
struct CreateFileRequest;
struct CreateFineTuningJobRequest;
struct FineTuneMethod;
struct FineTuneSupervisedMethod;
struct FineTuneDPOMethod;
struct CreateImageEditRequest;
struct CreateImageRequest;
struct CreateImageVariationRequest;
struct CreateMessageRequest;
struct CreateModerationRequest;
struct CreateModerationResponse;
struct CreateRunRequest;
struct TruncationObject;
struct CreateSpeechRequest;
struct CreateThreadAndRunRequest;
struct CreateThreadRequest;
struct CreateTranscriptionRequest;
struct CreateTranscriptionResponseJson;
struct CreateTranscriptionResponseVerboseJson;
struct CreateTranslationRequest;
struct CreateTranslationResponseJson;
struct CreateTranslationResponseVerboseJson;
struct CreateUploadRequest;
struct CreateVectorStoreFileBatchRequest;
struct CreateVectorStoreFileRequest;
struct CreateVectorStoreRequest;
struct VectorStoreExpirationAfter;
struct DefaultProjectErrorResponse;
struct DeleteAssistantResponse;
struct DeleteFileResponse;
struct DeleteMessageResponse;
struct DeleteModelResponse;
struct DeleteThreadResponse;
struct DeleteVectorStoreFileResponse;
struct DeleteVectorStoreResponse;
struct DoneEvent;
struct Embedding;
struct Error;
struct ErrorEvent;
struct ErrorResponse;
struct FileSearchRankingOptions;
struct FineTuneChatCompletionRequestAssistantMessage;
struct FineTuneChatRequestInput;
struct FineTuneCompletionRequestInput;
struct FineTunePreferenceRequestInput;
struct FineTuningIntegration;
struct FineTuningJob;
struct FineTuningJobCheckpoint;
struct FineTuningJobEvent;
struct Image;
struct ImagesResponse;
struct Invite;
struct InviteDeleteResponse;
struct InviteListResponse;
struct InviteRequest;
struct ListAssistantsResponse;
struct ListAuditLogsResponse;
struct ListBatchesResponse;
struct ListFilesResponse;
struct ListFineTuningJobCheckpointsResponse;
struct ListFineTuningJobEventsResponse;
struct ListMessagesResponse;
struct ListModelsResponse;
struct ListPaginatedFineTuningJobsResponse;
struct ListRunStepsResponse;
struct ListRunsResponse;
struct ListThreadsResponse;
struct ListVectorStoreFilesResponse;
struct ListVectorStoresResponse;
struct MessageContentImageFileObject;
struct MessageContentImageUrlObject;
struct MessageContentRefusalObject;
struct MessageContentTextAnnotationsFileCitationObject;
struct MessageContentTextAnnotationsFilePathObject;
struct MessageContentTextObject;
struct MessageDeltaContentImageFileObject;
struct MessageDeltaContentImageUrlObject;
struct MessageDeltaContentRefusalObject;
struct MessageDeltaContentTextAnnotationsFileCitationObject;
struct MessageDeltaContentTextAnnotationsFilePathObject;
struct MessageDeltaContentTextObject;
struct MessageDeltaObject;
struct MessageObject;
struct MessageRequestContentTextObject;
struct MessageStreamEvent;
struct Model;
struct ModifyAssistantRequest;
struct ModifyMessageRequest;
struct ModifyRunRequest;
struct ModifyThreadRequest;
struct OpenAIFile;
struct OtherChunkingStrategyResponseParam;
struct PredictionContent;
struct Project;
struct ProjectApiKey;
struct ProjectApiKeyDeleteResponse;
struct ProjectApiKeyListResponse;
struct ProjectCreateRequest;
struct ProjectListResponse;
struct ProjectRateLimit;
struct ProjectRateLimitListResponse;
struct ProjectRateLimitUpdateRequest;
struct ProjectServiceAccount;
struct ProjectServiceAccountApiKey;
struct ProjectServiceAccountCreateRequest;
struct ProjectServiceAccountCreateResponse;
struct ProjectServiceAccountDeleteResponse;
struct ProjectServiceAccountListResponse;
struct ProjectUpdateRequest;
struct ProjectUser;
struct ProjectUserCreateRequest;
struct ProjectUserDeleteResponse;
struct ProjectUserListResponse;
struct ProjectUserUpdateRequest;
struct RealtimeClientEventConversationItemCreate;
struct RealtimeConversationItem;
struct RealtimeClientEventConversationItemDelete;
struct RealtimeClientEventConversationItemTruncate;
struct RealtimeClientEventInputAudioBufferAppend;
struct RealtimeClientEventInputAudioBufferClear;
struct RealtimeClientEventInputAudioBufferCommit;
struct RealtimeClientEventResponseCancel;
struct RealtimeClientEventResponseCreate;
struct RealtimeResponseCreateParams;
struct RealtimeClientEventSessionUpdate;
struct RealtimeSessionCreateRequest;
struct RealtimeResponse;
struct RealtimeServerEventConversationCreated;
struct RealtimeServerEventConversationItemCreated;
struct RealtimeServerEventConversationItemDeleted;
struct RealtimeServerEventConversationItemInputAudioTranscriptionCompleted;
struct RealtimeServerEventConversationItemInputAudioTranscriptionFailed;
struct RealtimeServerEventConversationItemTruncated;
struct RealtimeServerEventError;
struct RealtimeServerEventInputAudioBufferCleared;
struct RealtimeServerEventInputAudioBufferCommitted;
struct RealtimeServerEventInputAudioBufferSpeechStarted;
struct RealtimeServerEventInputAudioBufferSpeechStopped;
struct RealtimeServerEventRateLimitsUpdated;
struct RealtimeServerEventResponseAudioDelta;
struct RealtimeServerEventResponseAudioDone;
struct RealtimeServerEventResponseAudioTranscriptDelta;
struct RealtimeServerEventResponseAudioTranscriptDone;
struct RealtimeServerEventResponseContentPartAdded;
struct RealtimeServerEventResponseContentPartDone;
struct RealtimeServerEventResponseCreated;
struct RealtimeServerEventResponseDone;
struct RealtimeServerEventResponseFunctionCallArgumentsDelta;
struct RealtimeServerEventResponseFunctionCallArgumentsDone;
struct RealtimeServerEventResponseOutputItemAdded;
struct RealtimeServerEventResponseOutputItemDone;
struct RealtimeServerEventResponseTextDelta;
struct RealtimeServerEventResponseTextDone;
struct RealtimeServerEventSessionCreated;
struct RealtimeSession;
struct RealtimeServerEventSessionUpdated;
struct RealtimeSessionCreateResponse;
struct ResponseFormatJsonObject;
struct ResponseFormatJsonSchema;
struct ResponseFormatJsonSchemaSchema;
struct ResponseFormatText;
struct RunCompletionUsage;
struct RunObject;
struct RunStepCompletionUsage;
struct RunStepDeltaObject;
struct RunStepDeltaStepDetailsMessageCreationObject;
struct RunStepDeltaStepDetailsToolCallsCodeObject;
struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObject;
struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject;
struct RunStepDeltaStepDetailsToolCallsFileSearchObject;
struct RunStepDeltaStepDetailsToolCallsFunctionObject;
struct RunStepDeltaStepDetailsToolCallsObject;
struct RunStepDetailsMessageCreationObject;
struct RunStepDetailsToolCallsCodeObject;
struct RunStepDetailsToolCallsCodeOutputImageObject;
struct RunStepDetailsToolCallsCodeOutputLogsObject;
struct RunStepDetailsToolCallsFileSearchObject;
struct RunStepDetailsToolCallsFileSearchRankingOptionsObject;
struct RunStepDetailsToolCallsFileSearchResultObject;
struct RunStepDetailsToolCallsFunctionObject;
struct RunStepDetailsToolCallsObject;
struct RunStepObject;
struct RunStepStreamEvent;
struct RunStreamEvent;
struct RunToolCallObject;
struct StaticChunkingStrategy;
struct StaticChunkingStrategyRequestParam;
struct StaticChunkingStrategyResponseParam;
struct SubmitToolOutputsRunRequest;
struct ThreadObject;
struct ThreadStreamEvent;
struct TranscriptionSegment;
struct TranscriptionWord;
struct UpdateVectorStoreRequest;
struct Upload;
struct UploadPart;
struct UsageAudioSpeechesResult;
struct UsageAudioTranscriptionsResult;
struct UsageCodeInterpreterSessionsResult;
struct UsageCompletionsResult;
struct UsageEmbeddingsResult;
struct UsageImagesResult;
struct UsageModerationsResult;
struct UsageResponse;
struct UsageTimeBucket;
struct UsageVectorStoresResult;
struct User;
struct UserDeleteResponse;
struct UserListResponse;
struct UserRoleUpdateRequest;
struct VectorStoreFileBatchObject;
struct VectorStoreFileObject;
struct VectorStoreObject;

// Struct definitions:
struct AddUploadPartRequest {
    UNKNOWN_TYPE data;
};

struct AdminApiKey {
    std::string object;
    std::string id;
    std::string name;
    std::string redacted_value;
    std::string value;
    long long created_at;
    UNKNOWN_TYPE owner;
};

struct ApiKeyList {
    std::string object;
    std::vector<AdminApiKey> data;
    bool has_more;
    std::string first_id;
    std::string last_id;
};

struct AssistantsApiResponseFormatOption {
};

struct AssistantObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string name;
    std::string description;
    std::string model;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
    AssistantsApiResponseFormatOption response_format;
};

struct AssistantStreamEvent {
};

struct AssistantToolsCode {
    std::string  // or generate an enum type;
};

struct AssistantToolsFileSearch {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

struct AssistantToolsFileSearchTypeOnly {
    std::string  // or generate an enum type;
};

struct FunctionParameters {
};

struct FunctionObject {
    std::string description;
    std::string name;
    FunctionParameters parameters;
    bool strict;
};

struct AssistantToolsFunction {
    std::string  // or generate an enum type;
    FunctionObject function;
};

struct AssistantsApiToolChoiceOption {
};

struct AssistantsNamedToolChoice {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct AudioResponseFormat {
};

struct AuditLogEventType {
};

struct AuditLogActorUser {
    std::string id;
    std::string email;
};

struct AuditLogActorSession {
    AuditLogActorUser user;
    std::string ip_address;
};

struct AuditLogActorServiceAccount {
    std::string id;
};

struct AuditLogActorApiKey {
    std::string id;
    std::string  // or generate an enum type;
    AuditLogActorUser user;
    AuditLogActorServiceAccount service_account;
};

struct AuditLogActor {
    std::string  // or generate an enum type;
    AuditLogActorSession session;
    AuditLogActorApiKey api_key;
};

struct AuditLog {
    std::string id;
    AuditLogEventType type;
    int effective_at;
    UNKNOWN_TYPE project;
    AuditLogActor actor;
    UNKNOWN_TYPE api_key.created;
    UNKNOWN_TYPE api_key.updated;
    UNKNOWN_TYPE api_key.deleted;
    UNKNOWN_TYPE invite.sent;
    UNKNOWN_TYPE invite.accepted;
    UNKNOWN_TYPE invite.deleted;
    UNKNOWN_TYPE login.failed;
    UNKNOWN_TYPE logout.failed;
    UNKNOWN_TYPE organization.updated;
    UNKNOWN_TYPE project.created;
    UNKNOWN_TYPE project.updated;
    UNKNOWN_TYPE project.archived;
    UNKNOWN_TYPE rate_limit.updated;
    UNKNOWN_TYPE rate_limit.deleted;
    UNKNOWN_TYPE service_account.created;
    UNKNOWN_TYPE service_account.updated;
    UNKNOWN_TYPE service_account.deleted;
    UNKNOWN_TYPE user.added;
    UNKNOWN_TYPE user.updated;
    UNKNOWN_TYPE user.deleted;
};

struct AutoChunkingStrategyRequestParam {
    std::string  // or generate an enum type;
};

struct Batch {
    std::string id;
    std::string  // or generate an enum object;
    std::string endpoint;
    UNKNOWN_TYPE errors;
    std::string input_file_id;
    std::string completion_window;
    std::string  // or generate an enum status;
    std::string output_file_id;
    std::string error_file_id;
    int created_at;
    int in_progress_at;
    int expires_at;
    int finalizing_at;
    int completed_at;
    int failed_at;
    int expired_at;
    int cancelling_at;
    int cancelled_at;
    UNKNOWN_TYPE request_counts;
    UNKNOWN_TYPE metadata;
};

struct BatchRequestInput {
    std::string custom_id;
    std::string  // or generate an enum method;
    std::string url;
};

struct BatchRequestOutput {
    std::string id;
    std::string custom_id;
    UNKNOWN_TYPE response;
    UNKNOWN_TYPE error;
};

struct CancelUploadRequest {
};

struct ChatCompletionFunctionCallOption {
    std::string name;
};

struct ChatCompletionFunctions {
    std::string description;
    std::string name;
    FunctionParameters parameters;
};

struct ChatCompletionMessageToolCall {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct ChatCompletionMessageToolCallChunk {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct ChatCompletionMessageToolCalls {
};

struct ChatCompletionModalities {
};

struct ChatCompletionNamedToolChoice {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct ChatCompletionRequestAssistantMessage {
    UNKNOWN_TYPE content;
    std::string refusal;
    std::string  // or generate an enum role;
    std::string name;
    UNKNOWN_TYPE audio;
    ChatCompletionMessageToolCalls tool_calls;
    UNKNOWN_TYPE function_call;
};

struct ChatCompletionRequestAssistantMessageContentPart {
};

struct ChatCompletionRequestDeveloperMessage {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

struct ChatCompletionRequestFunctionMessage {
    std::string  // or generate an enum role;
    std::string content;
    std::string name;
};

struct ChatCompletionRequestMessage {
};

struct ChatCompletionRequestMessageContentPartAudio {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE input_audio;
};

struct ChatCompletionRequestMessageContentPartImage {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

struct ChatCompletionRequestMessageContentPartRefusal {
    std::string  // or generate an enum type;
    std::string refusal;
};

struct ChatCompletionRequestMessageContentPartText {
    std::string  // or generate an enum type;
    std::string text;
};

struct ChatCompletionRequestSystemMessage {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

struct ChatCompletionRequestSystemMessageContentPart {
};

struct ChatCompletionRequestToolMessage {
    std::string  // or generate an enum role;
    UNKNOWN_TYPE content;
    std::string tool_call_id;
};

struct ChatCompletionRequestToolMessageContentPart {
};

struct ChatCompletionRequestUserMessage {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

struct ChatCompletionRequestUserMessageContentPart {
};

struct ChatCompletionResponseMessage {
    std::string content;
    std::string refusal;
    ChatCompletionMessageToolCalls tool_calls;
    std::string  // or generate an enum role;
    UNKNOWN_TYPE function_call;
    UNKNOWN_TYPE audio;
};

struct ChatCompletionRole {
};

struct ChatCompletionStreamOptions {
    bool include_usage;
};

struct ChatCompletionStreamResponseDelta {
    std::string content;
    UNKNOWN_TYPE function_call;
    std::vector<ChatCompletionMessageToolCallChunk> tool_calls;
    std::string  // or generate an enum role;
    std::string refusal;
};

struct ChatCompletionTokenLogprob {
    std::string token;
    double logprob;
    std::vector<int> bytes;
    std::vector<UNKNOWN_TYPE> top_logprobs;
};

struct ChatCompletionTool {
    std::string  // or generate an enum type;
    FunctionObject function;
};

struct ChatCompletionToolChoiceOption {
};

struct ChunkingStrategyRequestParam {
};

struct CompleteUploadRequest {
    std::vector<std::string> part_ids;
    std::string md5;
};

struct CompletionUsage {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
    UNKNOWN_TYPE completion_tokens_details;
    UNKNOWN_TYPE prompt_tokens_details;
};

struct CostsResult {
    std::string  // or generate an enum object;
    UNKNOWN_TYPE amount;
    std::string line_item;
    std::string project_id;
};

struct CreateAssistantRequest {
    UNKNOWN_TYPE model;
    std::string name;
    std::string description;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
    AssistantsApiResponseFormatOption response_format;
};

struct CreateChatCompletionFunctionResponse {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
    CompletionUsage usage;
};

struct CreateChatCompletionImageResponse {
};

struct ParallelToolCalls {
};

struct CreateChatCompletionRequest {
    std::vector<ChatCompletionRequestMessage> messages;
    UNKNOWN_TYPE model;
    bool store;
    std::string  // or generate an enum reasoning_effort;
    UNKNOWN_TYPE metadata;
    double frequency_penalty;
    UNKNOWN_TYPE logit_bias;
    bool logprobs;
    int top_logprobs;
    int max_tokens;
    int max_completion_tokens;
    int n;
    ChatCompletionModalities modalities;
    UNKNOWN_TYPE prediction;
    UNKNOWN_TYPE audio;
    double presence_penalty;
    UNKNOWN_TYPE response_format;
    int seed;
    std::string  // or generate an enum service_tier;
    UNKNOWN_TYPE stop;
    bool stream;
    ChatCompletionStreamOptions stream_options;
    double temperature;
    double top_p;
    std::vector<ChatCompletionTool> tools;
    ChatCompletionToolChoiceOption tool_choice;
    ParallelToolCalls parallel_tool_calls;
    std::string user;
    UNKNOWN_TYPE function_call;
    std::vector<ChatCompletionFunctions> functions;
};

struct CreateChatCompletionResponse {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string  // or generate an enum service_tier;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
    CompletionUsage usage;
};

struct CreateChatCompletionStreamResponse {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string  // or generate an enum service_tier;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE usage;
};

struct CreateCompletionRequest {
    UNKNOWN_TYPE model;
    UNKNOWN_TYPE prompt;
    int best_of;
    bool echo;
    double frequency_penalty;
    UNKNOWN_TYPE logit_bias;
    int logprobs;
    int max_tokens;
    int n;
    double presence_penalty;
    int seed;
    UNKNOWN_TYPE stop;
    bool stream;
    ChatCompletionStreamOptions stream_options;
    std::string suffix;
    double temperature;
    double top_p;
    std::string user;
};

struct CreateCompletionResponse {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
    CompletionUsage usage;
};

struct CreateEmbeddingRequest {
    UNKNOWN_TYPE input;
    UNKNOWN_TYPE model;
    std::string  // or generate an enum encoding_format;
    int dimensions;
    std::string user;
};

struct CreateEmbeddingResponse {
    std::vector<Embedding> data;
    std::string model;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE usage;
};

struct CreateFileRequest {
    UNKNOWN_TYPE file;
    std::string  // or generate an enum purpose;
};

struct FineTuneSupervisedMethod {
    UNKNOWN_TYPE hyperparameters;
};

struct FineTuneDPOMethod {
    UNKNOWN_TYPE hyperparameters;
};

struct FineTuneMethod {
    std::string  // or generate an enum type;
    FineTuneSupervisedMethod supervised;
    FineTuneDPOMethod dpo;
};

struct CreateFineTuningJobRequest {
    UNKNOWN_TYPE model;
    std::string training_file;
    UNKNOWN_TYPE hyperparameters;
    std::string suffix;
    std::string validation_file;
    std::vector<UNKNOWN_TYPE> integrations;
    int seed;
    FineTuneMethod method;
};

struct CreateImageEditRequest {
    UNKNOWN_TYPE image;
    std::string prompt;
    UNKNOWN_TYPE mask;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum size;
    std::string  // or generate an enum response_format;
    std::string user;
};

struct CreateImageRequest {
    std::string prompt;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum quality;
    std::string  // or generate an enum response_format;
    std::string  // or generate an enum size;
    std::string  // or generate an enum style;
    std::string user;
};

struct CreateImageVariationRequest {
    UNKNOWN_TYPE image;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum response_format;
    std::string  // or generate an enum size;
    std::string user;
};

struct CreateMessageRequest {
    std::string  // or generate an enum role;
    UNKNOWN_TYPE content;
    std::vector<UNKNOWN_TYPE> attachments;
    UNKNOWN_TYPE metadata;
};

struct CreateModerationRequest {
    UNKNOWN_TYPE input;
    UNKNOWN_TYPE model;
};

struct CreateModerationResponse {
    std::string id;
    std::string model;
    std::vector<UNKNOWN_TYPE> results;
};

struct TruncationObject {
    std::string  // or generate an enum type;
    int last_messages;
};

struct CreateRunRequest {
    std::string assistant_id;
    UNKNOWN_TYPE model;
    std::string instructions;
    std::string additional_instructions;
    std::vector<CreateMessageRequest> additional_messages;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
    bool stream;
    int max_prompt_tokens;
    int max_completion_tokens;
    TruncationObject truncation_strategy;
    AssistantsApiToolChoiceOption tool_choice;
    ParallelToolCalls parallel_tool_calls;
    AssistantsApiResponseFormatOption response_format;
};

struct CreateSpeechRequest {
    UNKNOWN_TYPE model;
    std::string input;
    std::string  // or generate an enum voice;
    std::string  // or generate an enum response_format;
    double speed;
};

struct CreateThreadRequest {
    std::vector<CreateMessageRequest> messages;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

struct CreateThreadAndRunRequest {
    std::string assistant_id;
    CreateThreadRequest thread;
    UNKNOWN_TYPE model;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
    bool stream;
    int max_prompt_tokens;
    int max_completion_tokens;
    TruncationObject truncation_strategy;
    AssistantsApiToolChoiceOption tool_choice;
    ParallelToolCalls parallel_tool_calls;
    AssistantsApiResponseFormatOption response_format;
};

struct CreateTranscriptionRequest {
    UNKNOWN_TYPE file;
    UNKNOWN_TYPE model;
    std::string language;
    std::string prompt;
    AudioResponseFormat response_format;
    double temperature;
    std::vector<std::string  // or generate an enum> timestamp_granularities[];
};

struct CreateTranscriptionResponseJson {
    std::string text;
};

struct CreateTranscriptionResponseVerboseJson {
    std::string language;
    std::string duration;
    std::string text;
    std::vector<TranscriptionWord> words;
    std::vector<TranscriptionSegment> segments;
};

struct CreateTranslationRequest {
    UNKNOWN_TYPE file;
    UNKNOWN_TYPE model;
    std::string prompt;
    AudioResponseFormat response_format;
    double temperature;
};

struct CreateTranslationResponseJson {
    std::string text;
};

struct CreateTranslationResponseVerboseJson {
    std::string language;
    std::string duration;
    std::string text;
    std::vector<TranscriptionSegment> segments;
};

struct CreateUploadRequest {
    std::string filename;
    std::string  // or generate an enum purpose;
    int bytes;
    std::string mime_type;
};

struct CreateVectorStoreFileBatchRequest {
    std::vector<std::string> file_ids;
    ChunkingStrategyRequestParam chunking_strategy;
};

struct CreateVectorStoreFileRequest {
    std::string file_id;
    ChunkingStrategyRequestParam chunking_strategy;
};

struct VectorStoreExpirationAfter {
    std::string  // or generate an enum anchor;
    int days;
};

struct CreateVectorStoreRequest {
    std::vector<std::string> file_ids;
    std::string name;
    VectorStoreExpirationAfter expires_after;
    UNKNOWN_TYPE chunking_strategy;
    UNKNOWN_TYPE metadata;
};

struct DefaultProjectErrorResponse {
    int code;
    std::string message;
};

struct DeleteAssistantResponse {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

struct DeleteFileResponse {
    std::string id;
    std::string  // or generate an enum object;
    bool deleted;
};

struct DeleteMessageResponse {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

struct DeleteModelResponse {
    std::string id;
    bool deleted;
    std::string object;
};

struct DeleteThreadResponse {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

struct DeleteVectorStoreFileResponse {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

struct DeleteVectorStoreResponse {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

struct DoneEvent {
    std::string  // or generate an enum event;
    std::string  // or generate an enum data;
};

struct Embedding {
    int index;
    std::vector<double> embedding;
    std::string  // or generate an enum object;
};

struct Error {
    std::string code;
    std::string message;
    std::string param;
    std::string type;
};

struct ErrorEvent {
    std::string  // or generate an enum event;
    Error data;
};

struct ErrorResponse {
    Error error;
};

struct FileSearchRankingOptions {
    std::string  // or generate an enum ranker;
    double score_threshold;
};

struct FineTuneChatCompletionRequestAssistantMessage {
};

struct FineTuneChatRequestInput {
    std::vector<UNKNOWN_TYPE> messages;
    std::vector<ChatCompletionTool> tools;
    ParallelToolCalls parallel_tool_calls;
    std::vector<ChatCompletionFunctions> functions;
};

struct FineTuneCompletionRequestInput {
    std::string prompt;
    std::string completion;
};

struct FineTunePreferenceRequestInput {
    UNKNOWN_TYPE input;
    std::vector<UNKNOWN_TYPE> preferred_completion;
    std::vector<UNKNOWN_TYPE> non_preferred_completion;
};

struct FineTuningIntegration {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE wandb;
};

struct FineTuningJob {
    std::string id;
    int created_at;
    UNKNOWN_TYPE error;
    std::string fine_tuned_model;
    int finished_at;
    UNKNOWN_TYPE hyperparameters;
    std::string model;
    std::string  // or generate an enum object;
    std::string organization_id;
    std::vector<std::string> result_files;
    std::string  // or generate an enum status;
    int trained_tokens;
    std::string training_file;
    std::string validation_file;
    std::vector<UNKNOWN_TYPE> integrations;
    int seed;
    int estimated_finish;
    FineTuneMethod method;
};

struct FineTuningJobCheckpoint {
    std::string id;
    int created_at;
    std::string fine_tuned_model_checkpoint;
    int step_number;
    UNKNOWN_TYPE metrics;
    std::string fine_tuning_job_id;
    std::string  // or generate an enum object;
};

struct FineTuningJobEvent {
    std::string  // or generate an enum object;
    std::string id;
    int created_at;
    std::string  // or generate an enum level;
    std::string message;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE data;
};

struct Image {
    std::string b64_json;
    std::string url;
    std::string revised_prompt;
};

struct ImagesResponse {
    int created;
    std::vector<Image> data;
};

struct Invite {
    std::string  // or generate an enum object;
    std::string id;
    std::string email;
    std::string  // or generate an enum role;
    std::string  // or generate an enum status;
    int invited_at;
    int expires_at;
    int accepted_at;
    std::vector<UNKNOWN_TYPE> projects;
};

struct InviteDeleteResponse {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

struct InviteListResponse {
    std::string  // or generate an enum object;
    std::vector<Invite> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct InviteRequest {
    std::string email;
    std::string  // or generate an enum role;
    std::vector<UNKNOWN_TYPE> projects;
};

struct ListAssistantsResponse {
    std::string object;
    std::vector<AssistantObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListAuditLogsResponse {
    std::string  // or generate an enum object;
    std::vector<AuditLog> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListBatchesResponse {
    std::vector<Batch> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
    std::string  // or generate an enum object;
};

struct ListFilesResponse {
    std::string object;
    std::vector<OpenAIFile> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListFineTuningJobCheckpointsResponse {
    std::vector<FineTuningJobCheckpoint> data;
    std::string  // or generate an enum object;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListFineTuningJobEventsResponse {
    std::vector<FineTuningJobEvent> data;
    std::string  // or generate an enum object;
};

struct ListMessagesResponse {
    std::string object;
    std::vector<MessageObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListModelsResponse {
    std::string  // or generate an enum object;
    std::vector<Model> data;
};

struct ListPaginatedFineTuningJobsResponse {
    std::vector<FineTuningJob> data;
    bool has_more;
    std::string  // or generate an enum object;
};

struct ListRunStepsResponse {
    std::string object;
    std::vector<RunStepObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListRunsResponse {
    std::string object;
    std::vector<RunObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListThreadsResponse {
    std::string object;
    std::vector<ThreadObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListVectorStoreFilesResponse {
    std::string object;
    std::vector<VectorStoreFileObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ListVectorStoresResponse {
    std::string object;
    std::vector<VectorStoreObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct MessageContentImageFileObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_file;
};

struct MessageContentImageUrlObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

struct MessageContentRefusalObject {
    std::string  // or generate an enum type;
    std::string refusal;
};

struct MessageContentTextAnnotationsFileCitationObject {
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_citation;
    int start_index;
    int end_index;
};

struct MessageContentTextAnnotationsFilePathObject {
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_path;
    int start_index;
    int end_index;
};

struct MessageContentTextObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE text;
};

struct MessageDeltaContentImageFileObject {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_file;
};

struct MessageDeltaContentImageUrlObject {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

struct MessageDeltaContentRefusalObject {
    int index;
    std::string  // or generate an enum type;
    std::string refusal;
};

struct MessageDeltaContentTextAnnotationsFileCitationObject {
    int index;
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_citation;
    int start_index;
    int end_index;
};

struct MessageDeltaContentTextAnnotationsFilePathObject {
    int index;
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_path;
    int start_index;
    int end_index;
};

struct MessageDeltaContentTextObject {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE text;
};

struct MessageDeltaObject {
    std::string id;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE delta;
};

struct MessageObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string thread_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE incomplete_details;
    int completed_at;
    int incomplete_at;
    std::string  // or generate an enum role;
    std::vector<UNKNOWN_TYPE> content;
    std::string assistant_id;
    std::string run_id;
    std::vector<UNKNOWN_TYPE> attachments;
    UNKNOWN_TYPE metadata;
};

struct MessageRequestContentTextObject {
    std::string  // or generate an enum type;
    std::string text;
};

struct MessageStreamEvent {
};

struct Model {
    std::string id;
    int created;
    std::string  // or generate an enum object;
    std::string owned_by;
};

struct ModifyAssistantRequest {
    UNKNOWN_TYPE model;
    std::string name;
    std::string description;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
    AssistantsApiResponseFormatOption response_format;
};

struct ModifyMessageRequest {
    UNKNOWN_TYPE metadata;
};

struct ModifyRunRequest {
    UNKNOWN_TYPE metadata;
};

struct ModifyThreadRequest {
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

struct OpenAIFile {
    std::string id;
    int bytes;
    int created_at;
    std::string filename;
    std::string  // or generate an enum object;
    std::string  // or generate an enum purpose;
    std::string  // or generate an enum status;
    std::string status_details;
};

struct OtherChunkingStrategyResponseParam {
    std::string  // or generate an enum type;
};

struct PredictionContent {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE content;
};

struct Project {
    std::string id;
    std::string  // or generate an enum object;
    std::string name;
    int created_at;
    int archived_at;
    std::string  // or generate an enum status;
};

struct ProjectApiKey {
    std::string  // or generate an enum object;
    std::string redacted_value;
    std::string name;
    int created_at;
    std::string id;
    UNKNOWN_TYPE owner;
};

struct ProjectApiKeyDeleteResponse {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

struct ProjectApiKeyListResponse {
    std::string  // or generate an enum object;
    std::vector<ProjectApiKey> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ProjectCreateRequest {
    std::string name;
};

struct ProjectListResponse {
    std::string  // or generate an enum object;
    std::vector<Project> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ProjectRateLimit {
    std::string  // or generate an enum object;
    std::string id;
    std::string model;
    int max_requests_per_1_minute;
    int max_tokens_per_1_minute;
    int max_images_per_1_minute;
    int max_audio_megabytes_per_1_minute;
    int max_requests_per_1_day;
    int batch_1_day_max_input_tokens;
};

struct ProjectRateLimitListResponse {
    std::string  // or generate an enum object;
    std::vector<ProjectRateLimit> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ProjectRateLimitUpdateRequest {
    int max_requests_per_1_minute;
    int max_tokens_per_1_minute;
    int max_images_per_1_minute;
    int max_audio_megabytes_per_1_minute;
    int max_requests_per_1_day;
    int batch_1_day_max_input_tokens;
};

struct ProjectServiceAccount {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string  // or generate an enum role;
    int created_at;
};

struct ProjectServiceAccountApiKey {
    std::string  // or generate an enum object;
    std::string value;
    std::string name;
    int created_at;
    std::string id;
};

struct ProjectServiceAccountCreateRequest {
    std::string name;
};

struct ProjectServiceAccountCreateResponse {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string  // or generate an enum role;
    int created_at;
    ProjectServiceAccountApiKey api_key;
};

struct ProjectServiceAccountDeleteResponse {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

struct ProjectServiceAccountListResponse {
    std::string  // or generate an enum object;
    std::vector<ProjectServiceAccount> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ProjectUpdateRequest {
    std::string name;
};

struct ProjectUser {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string email;
    std::string  // or generate an enum role;
    int added_at;
};

struct ProjectUserCreateRequest {
    std::string user_id;
    std::string  // or generate an enum role;
};

struct ProjectUserDeleteResponse {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

struct ProjectUserListResponse {
    std::string object;
    std::vector<ProjectUser> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct ProjectUserUpdateRequest {
    std::string  // or generate an enum role;
};

struct RealtimeConversationItem {
    std::string id;
    std::string  // or generate an enum type;
    std::string  // or generate an enum object;
    std::string  // or generate an enum status;
    std::string  // or generate an enum role;
    std::vector<UNKNOWN_TYPE> content;
    std::string call_id;
    std::string name;
    std::string arguments;
    std::string output;
};

struct RealtimeClientEventConversationItemCreate {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
    RealtimeConversationItem item;
};

struct RealtimeClientEventConversationItemDelete {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
};

struct RealtimeClientEventConversationItemTruncate {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    int audio_end_ms;
};

struct RealtimeClientEventInputAudioBufferAppend {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string audio;
};

struct RealtimeClientEventInputAudioBufferClear {
    std::string event_id;
    std::string  // or generate an enum type;
};

struct RealtimeClientEventInputAudioBufferCommit {
    std::string event_id;
    std::string  // or generate an enum type;
};

struct RealtimeClientEventResponseCancel {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
};

struct RealtimeResponseCreateParams {
    std::vector<std::string  // or generate an enum> modalities;
    std::string instructions;
    std::string  // or generate an enum voice;
    std::string  // or generate an enum output_audio_format;
    std::vector<UNKNOWN_TYPE> tools;
    std::string tool_choice;
    double temperature;
    UNKNOWN_TYPE max_response_output_tokens;
    UNKNOWN_TYPE conversation;
    UNKNOWN_TYPE metadata;
    std::vector<RealtimeConversationItem> input;
};

struct RealtimeClientEventResponseCreate {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeResponseCreateParams response;
};

struct RealtimeSessionCreateRequest {
    UNKNOWN_TYPE modalities;
    std::string  // or generate an enum model;
    std::string instructions;
    std::string  // or generate an enum voice;
    std::string  // or generate an enum input_audio_format;
    std::string  // or generate an enum output_audio_format;
    UNKNOWN_TYPE input_audio_transcription;
    UNKNOWN_TYPE turn_detection;
    std::vector<UNKNOWN_TYPE> tools;
    std::string tool_choice;
    double temperature;
    UNKNOWN_TYPE max_response_output_tokens;
};

struct RealtimeClientEventSessionUpdate {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeSessionCreateRequest session;
};

struct RealtimeResponse {
    std::string id;
    std::string  // or generate an enum object;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE status_details;
    std::vector<RealtimeConversationItem> output;
    UNKNOWN_TYPE metadata;
    UNKNOWN_TYPE usage;
};

struct RealtimeServerEventConversationCreated {
    std::string event_id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE conversation;
};

struct RealtimeServerEventConversationItemCreated {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
    RealtimeConversationItem item;
};

struct RealtimeServerEventConversationItemDeleted {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
};

struct RealtimeServerEventConversationItemInputAudioTranscriptionCompleted {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    std::string transcript;
};

struct RealtimeServerEventConversationItemInputAudioTranscriptionFailed {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    UNKNOWN_TYPE error;
};

struct RealtimeServerEventConversationItemTruncated {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    int audio_end_ms;
};

struct RealtimeServerEventError {
    std::string event_id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE error;
};

struct RealtimeServerEventInputAudioBufferCleared {
    std::string event_id;
    std::string  // or generate an enum type;
};

struct RealtimeServerEventInputAudioBufferCommitted {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
    std::string item_id;
};

struct RealtimeServerEventInputAudioBufferSpeechStarted {
    std::string event_id;
    std::string  // or generate an enum type;
    int audio_start_ms;
    std::string item_id;
};

struct RealtimeServerEventInputAudioBufferSpeechStopped {
    std::string event_id;
    std::string  // or generate an enum type;
    int audio_end_ms;
    std::string item_id;
};

struct RealtimeServerEventRateLimitsUpdated {
    std::string event_id;
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> rate_limits;
};

struct RealtimeServerEventResponseAudioDelta {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

struct RealtimeServerEventResponseAudioDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
};

struct RealtimeServerEventResponseAudioTranscriptDelta {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

struct RealtimeServerEventResponseAudioTranscriptDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string transcript;
};

struct RealtimeServerEventResponseContentPartAdded {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    UNKNOWN_TYPE part;
};

struct RealtimeServerEventResponseContentPartDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    UNKNOWN_TYPE part;
};

struct RealtimeServerEventResponseCreated {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeResponse response;
};

struct RealtimeServerEventResponseDone {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeResponse response;
};

struct RealtimeServerEventResponseFunctionCallArgumentsDelta {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    std::string call_id;
    std::string delta;
};

struct RealtimeServerEventResponseFunctionCallArgumentsDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    std::string call_id;
    std::string arguments;
};

struct RealtimeServerEventResponseOutputItemAdded {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    int output_index;
    RealtimeConversationItem item;
};

struct RealtimeServerEventResponseOutputItemDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    int output_index;
    RealtimeConversationItem item;
};

struct RealtimeServerEventResponseTextDelta {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

struct RealtimeServerEventResponseTextDone {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string text;
};

struct RealtimeSession {
    std::string id;
    UNKNOWN_TYPE modalities;
    UNKNOWN_TYPE model;
    std::string instructions;
    std::string  // or generate an enum voice;
    std::string  // or generate an enum input_audio_format;
    std::string  // or generate an enum output_audio_format;
    UNKNOWN_TYPE input_audio_transcription;
    UNKNOWN_TYPE turn_detection;
    std::vector<UNKNOWN_TYPE> tools;
    std::string tool_choice;
    double temperature;
    UNKNOWN_TYPE max_response_output_tokens;
};

struct RealtimeServerEventSessionCreated {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeSession session;
};

struct RealtimeServerEventSessionUpdated {
    std::string event_id;
    std::string  // or generate an enum type;
    RealtimeSession session;
};

struct RealtimeSessionCreateResponse {
    UNKNOWN_TYPE client_secret;
    UNKNOWN_TYPE modalities;
    std::string instructions;
    std::string  // or generate an enum voice;
    std::string input_audio_format;
    std::string output_audio_format;
    UNKNOWN_TYPE input_audio_transcription;
    UNKNOWN_TYPE turn_detection;
    std::vector<UNKNOWN_TYPE> tools;
    std::string tool_choice;
    double temperature;
    UNKNOWN_TYPE max_response_output_tokens;
};

struct ResponseFormatJsonObject {
    std::string  // or generate an enum type;
};

struct ResponseFormatJsonSchema {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE json_schema;
};

struct ResponseFormatJsonSchemaSchema {
};

struct ResponseFormatText {
    std::string  // or generate an enum type;
};

struct RunCompletionUsage {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
};

struct RunObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string thread_id;
    std::string assistant_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE required_action;
    UNKNOWN_TYPE last_error;
    int expires_at;
    int started_at;
    int cancelled_at;
    int failed_at;
    int completed_at;
    UNKNOWN_TYPE incomplete_details;
    std::string model;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE metadata;
    RunCompletionUsage usage;
    double temperature;
    double top_p;
    int max_prompt_tokens;
    int max_completion_tokens;
    TruncationObject truncation_strategy;
    AssistantsApiToolChoiceOption tool_choice;
    ParallelToolCalls parallel_tool_calls;
    AssistantsApiResponseFormatOption response_format;
};

struct RunStepCompletionUsage {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
};

struct RunStepDeltaObject {
    std::string id;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE delta;
};

struct RunStepDeltaStepDetailsMessageCreationObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE message_creation;
};

struct RunStepDeltaStepDetailsToolCallsCodeObject {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE code_interpreter;
};

struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObject {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image;
};

struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject {
    int index;
    std::string  // or generate an enum type;
    std::string logs;
};

struct RunStepDeltaStepDetailsToolCallsFileSearchObject {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

struct RunStepDeltaStepDetailsToolCallsFunctionObject {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct RunStepDeltaStepDetailsToolCallsObject {
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> tool_calls;
};

struct RunStepDetailsMessageCreationObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE message_creation;
};

struct RunStepDetailsToolCallsCodeObject {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE code_interpreter;
};

struct RunStepDetailsToolCallsCodeOutputImageObject {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image;
};

struct RunStepDetailsToolCallsCodeOutputLogsObject {
    std::string  // or generate an enum type;
    std::string logs;
};

struct RunStepDetailsToolCallsFileSearchObject {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

struct RunStepDetailsToolCallsFileSearchRankingOptionsObject {
    std::string  // or generate an enum ranker;
    double score_threshold;
};

struct RunStepDetailsToolCallsFileSearchResultObject {
    std::string file_id;
    std::string file_name;
    double score;
    std::vector<UNKNOWN_TYPE> content;
};

struct RunStepDetailsToolCallsFunctionObject {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct RunStepDetailsToolCallsObject {
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> tool_calls;
};

struct RunStepObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string assistant_id;
    std::string thread_id;
    std::string run_id;
    std::string  // or generate an enum type;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE step_details;
    UNKNOWN_TYPE last_error;
    int expired_at;
    int cancelled_at;
    int failed_at;
    int completed_at;
    UNKNOWN_TYPE metadata;
    RunStepCompletionUsage usage;
};

struct RunStepStreamEvent {
};

struct RunStreamEvent {
};

struct RunToolCallObject {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

struct StaticChunkingStrategy {
    int max_chunk_size_tokens;
    int chunk_overlap_tokens;
};

struct StaticChunkingStrategyRequestParam {
    std::string  // or generate an enum type;
    StaticChunkingStrategy static;
};

struct StaticChunkingStrategyResponseParam {
    std::string  // or generate an enum type;
    StaticChunkingStrategy static;
};

struct SubmitToolOutputsRunRequest {
    std::vector<UNKNOWN_TYPE> tool_outputs;
    bool stream;
};

struct ThreadObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

struct ThreadStreamEvent {
};

struct TranscriptionSegment {
    int id;
    int seek;
    UNKNOWN_TYPE start;
    UNKNOWN_TYPE end;
    std::string text;
    std::vector<int> tokens;
    UNKNOWN_TYPE temperature;
    UNKNOWN_TYPE avg_logprob;
    UNKNOWN_TYPE compression_ratio;
    UNKNOWN_TYPE no_speech_prob;
};

struct TranscriptionWord {
    std::string word;
    UNKNOWN_TYPE start;
    UNKNOWN_TYPE end;
};

struct UpdateVectorStoreRequest {
    std::string name;
    VectorStoreExpirationAfter expires_after;
    UNKNOWN_TYPE metadata;
};

struct Upload {
    std::string id;
    int created_at;
    std::string filename;
    int bytes;
    std::string purpose;
    std::string  // or generate an enum status;
    int expires_at;
    std::string  // or generate an enum object;
    OpenAIFile file;
};

struct UploadPart {
    std::string id;
    int created_at;
    std::string upload_id;
    std::string  // or generate an enum object;
};

struct UsageAudioSpeechesResult {
    std::string  // or generate an enum object;
    int characters;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

struct UsageAudioTranscriptionsResult {
    std::string  // or generate an enum object;
    int seconds;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

struct UsageCodeInterpreterSessionsResult {
    std::string  // or generate an enum object;
    int sessions;
    std::string project_id;
};

struct UsageCompletionsResult {
    std::string  // or generate an enum object;
    int input_tokens;
    int input_cached_tokens;
    int output_tokens;
    int input_audio_tokens;
    int output_audio_tokens;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
    bool batch;
};

struct UsageEmbeddingsResult {
    std::string  // or generate an enum object;
    int input_tokens;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

struct UsageImagesResult {
    std::string  // or generate an enum object;
    int images;
    int num_model_requests;
    std::string source;
    std::string size;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

struct UsageModerationsResult {
    std::string  // or generate an enum object;
    int input_tokens;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

struct UsageResponse {
    std::string  // or generate an enum object;
    std::vector<UsageTimeBucket> data;
    bool has_more;
    std::string next_page;
};

struct UsageTimeBucket {
    std::string  // or generate an enum object;
    int start_time;
    int end_time;
    std::vector<UNKNOWN_TYPE> result;
};

struct UsageVectorStoresResult {
    std::string  // or generate an enum object;
    int usage_bytes;
    std::string project_id;
};

struct User {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string email;
    std::string  // or generate an enum role;
    int added_at;
};

struct UserDeleteResponse {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

struct UserListResponse {
    std::string  // or generate an enum object;
    std::vector<User> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

struct UserRoleUpdateRequest {
    std::string  // or generate an enum role;
};

struct VectorStoreFileBatchObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string vector_store_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE file_counts;
};

struct VectorStoreFileObject {
    std::string id;
    std::string  // or generate an enum object;
    int usage_bytes;
    int created_at;
    std::string vector_store_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE last_error;
    UNKNOWN_TYPE chunking_strategy;
};

struct VectorStoreObject {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string name;
    int usage_bytes;
    UNKNOWN_TYPE file_counts;
    std::string  // or generate an enum status;
    VectorStoreExpirationAfter expires_after;
    int expires_at;
    int last_active_at;
    UNKNOWN_TYPE metadata;
};
