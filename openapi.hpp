#pragma once
#include <string>
#include <vector>
#include <optional>

// If we ever generate a fallback or unknown type, this macro triggers a compile-time error.
#define UNKNOWN_TYPE static_assert(false, "Unknown schema type encountered!");

// Forward declarations:
template <typename Derived> struct AddUploadPartRequestBase;
struct AddUploadPartRequest;
template <typename Derived> struct AdminApiKeyBase;
struct AdminApiKey;
template <typename Derived> struct ApiKeyListBase;
struct ApiKeyList;
template <typename Derived> struct AssistantObjectBase;
struct AssistantObject;
template <typename Derived> struct AssistantsApiResponseFormatOptionBase;
struct AssistantsApiResponseFormatOption;
template <typename Derived> struct AssistantStreamEventBase;
struct AssistantStreamEvent;
template <typename Derived> struct AssistantToolsCodeBase;
struct AssistantToolsCode;
template <typename Derived> struct AssistantToolsFileSearchBase;
struct AssistantToolsFileSearch;
template <typename Derived> struct AssistantToolsFileSearchTypeOnlyBase;
struct AssistantToolsFileSearchTypeOnly;
template <typename Derived> struct AssistantToolsFunctionBase;
struct AssistantToolsFunction;
template <typename Derived> struct FunctionObjectBase;
struct FunctionObject;
template <typename Derived> struct FunctionParametersBase;
struct FunctionParameters;
template <typename Derived> struct AssistantsApiToolChoiceOptionBase;
struct AssistantsApiToolChoiceOption;
template <typename Derived> struct AssistantsNamedToolChoiceBase;
struct AssistantsNamedToolChoice;
template <typename Derived> struct AudioResponseFormatBase;
struct AudioResponseFormat;
template <typename Derived> struct AuditLogBase;
struct AuditLog;
template <typename Derived> struct AuditLogEventTypeBase;
struct AuditLogEventType;
template <typename Derived> struct AuditLogActorBase;
struct AuditLogActor;
template <typename Derived> struct AuditLogActorSessionBase;
struct AuditLogActorSession;
template <typename Derived> struct AuditLogActorUserBase;
struct AuditLogActorUser;
template <typename Derived> struct AuditLogActorApiKeyBase;
struct AuditLogActorApiKey;
template <typename Derived> struct AuditLogActorServiceAccountBase;
struct AuditLogActorServiceAccount;
template <typename Derived> struct AutoChunkingStrategyRequestParamBase;
struct AutoChunkingStrategyRequestParam;
template <typename Derived> struct BatchBase;
struct Batch;
template <typename Derived> struct BatchRequestInputBase;
struct BatchRequestInput;
template <typename Derived> struct BatchRequestOutputBase;
struct BatchRequestOutput;
template <typename Derived> struct CancelUploadRequestBase;
struct CancelUploadRequest;
template <typename Derived> struct ChatCompletionFunctionCallOptionBase;
struct ChatCompletionFunctionCallOption;
template <typename Derived> struct ChatCompletionFunctionsBase;
struct ChatCompletionFunctions;
template <typename Derived> struct ChatCompletionMessageToolCallBase;
struct ChatCompletionMessageToolCall;
template <typename Derived> struct ChatCompletionMessageToolCallChunkBase;
struct ChatCompletionMessageToolCallChunk;
template <typename Derived> struct ChatCompletionMessageToolCallsBase;
struct ChatCompletionMessageToolCalls;
template <typename Derived> struct ChatCompletionModalitiesBase;
struct ChatCompletionModalities;
template <typename Derived> struct ChatCompletionNamedToolChoiceBase;
struct ChatCompletionNamedToolChoice;
template <typename Derived> struct ChatCompletionRequestAssistantMessageBase;
struct ChatCompletionRequestAssistantMessage;
template <typename Derived> struct ChatCompletionRequestAssistantMessageContentPartBase;
struct ChatCompletionRequestAssistantMessageContentPart;
template <typename Derived> struct ChatCompletionRequestDeveloperMessageBase;
struct ChatCompletionRequestDeveloperMessage;
template <typename Derived> struct ChatCompletionRequestFunctionMessageBase;
struct ChatCompletionRequestFunctionMessage;
template <typename Derived> struct ChatCompletionRequestMessageBase;
struct ChatCompletionRequestMessage;
template <typename Derived> struct ChatCompletionRequestMessageContentPartAudioBase;
struct ChatCompletionRequestMessageContentPartAudio;
template <typename Derived> struct ChatCompletionRequestMessageContentPartImageBase;
struct ChatCompletionRequestMessageContentPartImage;
template <typename Derived> struct ChatCompletionRequestMessageContentPartRefusalBase;
struct ChatCompletionRequestMessageContentPartRefusal;
template <typename Derived> struct ChatCompletionRequestMessageContentPartTextBase;
struct ChatCompletionRequestMessageContentPartText;
template <typename Derived> struct ChatCompletionRequestSystemMessageBase;
struct ChatCompletionRequestSystemMessage;
template <typename Derived> struct ChatCompletionRequestSystemMessageContentPartBase;
struct ChatCompletionRequestSystemMessageContentPart;
template <typename Derived> struct ChatCompletionRequestToolMessageBase;
struct ChatCompletionRequestToolMessage;
template <typename Derived> struct ChatCompletionRequestToolMessageContentPartBase;
struct ChatCompletionRequestToolMessageContentPart;
template <typename Derived> struct ChatCompletionRequestUserMessageBase;
struct ChatCompletionRequestUserMessage;
template <typename Derived> struct ChatCompletionRequestUserMessageContentPartBase;
struct ChatCompletionRequestUserMessageContentPart;
template <typename Derived> struct ChatCompletionResponseMessageBase;
struct ChatCompletionResponseMessage;
template <typename Derived> struct ChatCompletionRoleBase;
struct ChatCompletionRole;
template <typename Derived> struct ChatCompletionStreamOptionsBase;
struct ChatCompletionStreamOptions;
template <typename Derived> struct ChatCompletionStreamResponseDeltaBase;
struct ChatCompletionStreamResponseDelta;
template <typename Derived> struct ChatCompletionTokenLogprobBase;
struct ChatCompletionTokenLogprob;
template <typename Derived> struct ChatCompletionToolBase;
struct ChatCompletionTool;
template <typename Derived> struct ChatCompletionToolChoiceOptionBase;
struct ChatCompletionToolChoiceOption;
template <typename Derived> struct ChunkingStrategyRequestParamBase;
struct ChunkingStrategyRequestParam;
template <typename Derived> struct CompleteUploadRequestBase;
struct CompleteUploadRequest;
template <typename Derived> struct CompletionUsageBase;
struct CompletionUsage;
template <typename Derived> struct CostsResultBase;
struct CostsResult;
template <typename Derived> struct CreateAssistantRequestBase;
struct CreateAssistantRequest;
template <typename Derived> struct CreateChatCompletionFunctionResponseBase;
struct CreateChatCompletionFunctionResponse;
template <typename Derived> struct CreateChatCompletionImageResponseBase;
struct CreateChatCompletionImageResponse;
template <typename Derived> struct CreateChatCompletionRequestBase;
struct CreateChatCompletionRequest;
template <typename Derived> struct ParallelToolCallsBase;
struct ParallelToolCalls;
template <typename Derived> struct CreateChatCompletionResponseBase;
struct CreateChatCompletionResponse;
template <typename Derived> struct CreateChatCompletionStreamResponseBase;
struct CreateChatCompletionStreamResponse;
template <typename Derived> struct CreateCompletionRequestBase;
struct CreateCompletionRequest;
template <typename Derived> struct CreateCompletionResponseBase;
struct CreateCompletionResponse;
template <typename Derived> struct CreateEmbeddingRequestBase;
struct CreateEmbeddingRequest;
template <typename Derived> struct CreateEmbeddingResponseBase;
struct CreateEmbeddingResponse;
template <typename Derived> struct CreateFileRequestBase;
struct CreateFileRequest;
template <typename Derived> struct CreateFineTuningJobRequestBase;
struct CreateFineTuningJobRequest;
template <typename Derived> struct FineTuneMethodBase;
struct FineTuneMethod;
template <typename Derived> struct FineTuneSupervisedMethodBase;
struct FineTuneSupervisedMethod;
template <typename Derived> struct FineTuneDPOMethodBase;
struct FineTuneDPOMethod;
template <typename Derived> struct CreateImageEditRequestBase;
struct CreateImageEditRequest;
template <typename Derived> struct CreateImageRequestBase;
struct CreateImageRequest;
template <typename Derived> struct CreateImageVariationRequestBase;
struct CreateImageVariationRequest;
template <typename Derived> struct CreateMessageRequestBase;
struct CreateMessageRequest;
template <typename Derived> struct CreateModerationRequestBase;
struct CreateModerationRequest;
template <typename Derived> struct CreateModerationResponseBase;
struct CreateModerationResponse;
template <typename Derived> struct CreateRunRequestBase;
struct CreateRunRequest;
template <typename Derived> struct TruncationObjectBase;
struct TruncationObject;
template <typename Derived> struct CreateSpeechRequestBase;
struct CreateSpeechRequest;
template <typename Derived> struct CreateThreadAndRunRequestBase;
struct CreateThreadAndRunRequest;
template <typename Derived> struct CreateThreadRequestBase;
struct CreateThreadRequest;
template <typename Derived> struct CreateTranscriptionRequestBase;
struct CreateTranscriptionRequest;
template <typename Derived> struct CreateTranscriptionResponseJsonBase;
struct CreateTranscriptionResponseJson;
template <typename Derived> struct CreateTranscriptionResponseVerboseJsonBase;
struct CreateTranscriptionResponseVerboseJson;
template <typename Derived> struct CreateTranslationRequestBase;
struct CreateTranslationRequest;
template <typename Derived> struct CreateTranslationResponseJsonBase;
struct CreateTranslationResponseJson;
template <typename Derived> struct CreateTranslationResponseVerboseJsonBase;
struct CreateTranslationResponseVerboseJson;
template <typename Derived> struct CreateUploadRequestBase;
struct CreateUploadRequest;
template <typename Derived> struct CreateVectorStoreFileBatchRequestBase;
struct CreateVectorStoreFileBatchRequest;
template <typename Derived> struct CreateVectorStoreFileRequestBase;
struct CreateVectorStoreFileRequest;
template <typename Derived> struct CreateVectorStoreRequestBase;
struct CreateVectorStoreRequest;
template <typename Derived> struct VectorStoreExpirationAfterBase;
struct VectorStoreExpirationAfter;
template <typename Derived> struct DefaultProjectErrorResponseBase;
struct DefaultProjectErrorResponse;
template <typename Derived> struct DeleteAssistantResponseBase;
struct DeleteAssistantResponse;
template <typename Derived> struct DeleteFileResponseBase;
struct DeleteFileResponse;
template <typename Derived> struct DeleteMessageResponseBase;
struct DeleteMessageResponse;
template <typename Derived> struct DeleteModelResponseBase;
struct DeleteModelResponse;
template <typename Derived> struct DeleteThreadResponseBase;
struct DeleteThreadResponse;
template <typename Derived> struct DeleteVectorStoreFileResponseBase;
struct DeleteVectorStoreFileResponse;
template <typename Derived> struct DeleteVectorStoreResponseBase;
struct DeleteVectorStoreResponse;
template <typename Derived> struct DoneEventBase;
struct DoneEvent;
template <typename Derived> struct EmbeddingBase;
struct Embedding;
template <typename Derived> struct ErrorBase;
struct Error;
template <typename Derived> struct ErrorEventBase;
struct ErrorEvent;
template <typename Derived> struct ErrorResponseBase;
struct ErrorResponse;
template <typename Derived> struct FileSearchRankingOptionsBase;
struct FileSearchRankingOptions;
template <typename Derived> struct FineTuneChatCompletionRequestAssistantMessageBase;
struct FineTuneChatCompletionRequestAssistantMessage;
template <typename Derived> struct FineTuneChatRequestInputBase;
struct FineTuneChatRequestInput;
template <typename Derived> struct FineTuneCompletionRequestInputBase;
struct FineTuneCompletionRequestInput;
template <typename Derived> struct FineTunePreferenceRequestInputBase;
struct FineTunePreferenceRequestInput;
template <typename Derived> struct FineTuningIntegrationBase;
struct FineTuningIntegration;
template <typename Derived> struct FineTuningJobBase;
struct FineTuningJob;
template <typename Derived> struct FineTuningJobCheckpointBase;
struct FineTuningJobCheckpoint;
template <typename Derived> struct FineTuningJobEventBase;
struct FineTuningJobEvent;
template <typename Derived> struct ImageBase;
struct Image;
template <typename Derived> struct ImagesResponseBase;
struct ImagesResponse;
template <typename Derived> struct InviteBase;
struct Invite;
template <typename Derived> struct InviteDeleteResponseBase;
struct InviteDeleteResponse;
template <typename Derived> struct InviteListResponseBase;
struct InviteListResponse;
template <typename Derived> struct InviteRequestBase;
struct InviteRequest;
template <typename Derived> struct ListAssistantsResponseBase;
struct ListAssistantsResponse;
template <typename Derived> struct ListAuditLogsResponseBase;
struct ListAuditLogsResponse;
template <typename Derived> struct ListBatchesResponseBase;
struct ListBatchesResponse;
template <typename Derived> struct ListFilesResponseBase;
struct ListFilesResponse;
template <typename Derived> struct ListFineTuningJobCheckpointsResponseBase;
struct ListFineTuningJobCheckpointsResponse;
template <typename Derived> struct ListFineTuningJobEventsResponseBase;
struct ListFineTuningJobEventsResponse;
template <typename Derived> struct ListMessagesResponseBase;
struct ListMessagesResponse;
template <typename Derived> struct ListModelsResponseBase;
struct ListModelsResponse;
template <typename Derived> struct ListPaginatedFineTuningJobsResponseBase;
struct ListPaginatedFineTuningJobsResponse;
template <typename Derived> struct ListRunStepsResponseBase;
struct ListRunStepsResponse;
template <typename Derived> struct ListRunsResponseBase;
struct ListRunsResponse;
template <typename Derived> struct ListThreadsResponseBase;
struct ListThreadsResponse;
template <typename Derived> struct ListVectorStoreFilesResponseBase;
struct ListVectorStoreFilesResponse;
template <typename Derived> struct ListVectorStoresResponseBase;
struct ListVectorStoresResponse;
template <typename Derived> struct MessageContentImageFileObjectBase;
struct MessageContentImageFileObject;
template <typename Derived> struct MessageContentImageUrlObjectBase;
struct MessageContentImageUrlObject;
template <typename Derived> struct MessageContentRefusalObjectBase;
struct MessageContentRefusalObject;
template <typename Derived> struct MessageContentTextAnnotationsFileCitationObjectBase;
struct MessageContentTextAnnotationsFileCitationObject;
template <typename Derived> struct MessageContentTextAnnotationsFilePathObjectBase;
struct MessageContentTextAnnotationsFilePathObject;
template <typename Derived> struct MessageContentTextObjectBase;
struct MessageContentTextObject;
template <typename Derived> struct MessageDeltaContentImageFileObjectBase;
struct MessageDeltaContentImageFileObject;
template <typename Derived> struct MessageDeltaContentImageUrlObjectBase;
struct MessageDeltaContentImageUrlObject;
template <typename Derived> struct MessageDeltaContentRefusalObjectBase;
struct MessageDeltaContentRefusalObject;
template <typename Derived> struct MessageDeltaContentTextAnnotationsFileCitationObjectBase;
struct MessageDeltaContentTextAnnotationsFileCitationObject;
template <typename Derived> struct MessageDeltaContentTextAnnotationsFilePathObjectBase;
struct MessageDeltaContentTextAnnotationsFilePathObject;
template <typename Derived> struct MessageDeltaContentTextObjectBase;
struct MessageDeltaContentTextObject;
template <typename Derived> struct MessageDeltaObjectBase;
struct MessageDeltaObject;
template <typename Derived> struct MessageObjectBase;
struct MessageObject;
template <typename Derived> struct MessageRequestContentTextObjectBase;
struct MessageRequestContentTextObject;
template <typename Derived> struct MessageStreamEventBase;
struct MessageStreamEvent;
template <typename Derived> struct ModelBase;
struct Model;
template <typename Derived> struct ModifyAssistantRequestBase;
struct ModifyAssistantRequest;
template <typename Derived> struct ModifyMessageRequestBase;
struct ModifyMessageRequest;
template <typename Derived> struct ModifyRunRequestBase;
struct ModifyRunRequest;
template <typename Derived> struct ModifyThreadRequestBase;
struct ModifyThreadRequest;
template <typename Derived> struct OpenAIFileBase;
struct OpenAIFile;
template <typename Derived> struct OtherChunkingStrategyResponseParamBase;
struct OtherChunkingStrategyResponseParam;
template <typename Derived> struct PredictionContentBase;
struct PredictionContent;
template <typename Derived> struct ProjectBase;
struct Project;
template <typename Derived> struct ProjectApiKeyBase;
struct ProjectApiKey;
template <typename Derived> struct ProjectApiKeyDeleteResponseBase;
struct ProjectApiKeyDeleteResponse;
template <typename Derived> struct ProjectApiKeyListResponseBase;
struct ProjectApiKeyListResponse;
template <typename Derived> struct ProjectCreateRequestBase;
struct ProjectCreateRequest;
template <typename Derived> struct ProjectListResponseBase;
struct ProjectListResponse;
template <typename Derived> struct ProjectRateLimitBase;
struct ProjectRateLimit;
template <typename Derived> struct ProjectRateLimitListResponseBase;
struct ProjectRateLimitListResponse;
template <typename Derived> struct ProjectRateLimitUpdateRequestBase;
struct ProjectRateLimitUpdateRequest;
template <typename Derived> struct ProjectServiceAccountBase;
struct ProjectServiceAccount;
template <typename Derived> struct ProjectServiceAccountApiKeyBase;
struct ProjectServiceAccountApiKey;
template <typename Derived> struct ProjectServiceAccountCreateRequestBase;
struct ProjectServiceAccountCreateRequest;
template <typename Derived> struct ProjectServiceAccountCreateResponseBase;
struct ProjectServiceAccountCreateResponse;
template <typename Derived> struct ProjectServiceAccountDeleteResponseBase;
struct ProjectServiceAccountDeleteResponse;
template <typename Derived> struct ProjectServiceAccountListResponseBase;
struct ProjectServiceAccountListResponse;
template <typename Derived> struct ProjectUpdateRequestBase;
struct ProjectUpdateRequest;
template <typename Derived> struct ProjectUserBase;
struct ProjectUser;
template <typename Derived> struct ProjectUserCreateRequestBase;
struct ProjectUserCreateRequest;
template <typename Derived> struct ProjectUserDeleteResponseBase;
struct ProjectUserDeleteResponse;
template <typename Derived> struct ProjectUserListResponseBase;
struct ProjectUserListResponse;
template <typename Derived> struct ProjectUserUpdateRequestBase;
struct ProjectUserUpdateRequest;
template <typename Derived> struct RealtimeClientEventConversationItemCreateBase;
struct RealtimeClientEventConversationItemCreate;
template <typename Derived> struct RealtimeConversationItemBase;
struct RealtimeConversationItem;
template <typename Derived> struct RealtimeClientEventConversationItemDeleteBase;
struct RealtimeClientEventConversationItemDelete;
template <typename Derived> struct RealtimeClientEventConversationItemTruncateBase;
struct RealtimeClientEventConversationItemTruncate;
template <typename Derived> struct RealtimeClientEventInputAudioBufferAppendBase;
struct RealtimeClientEventInputAudioBufferAppend;
template <typename Derived> struct RealtimeClientEventInputAudioBufferClearBase;
struct RealtimeClientEventInputAudioBufferClear;
template <typename Derived> struct RealtimeClientEventInputAudioBufferCommitBase;
struct RealtimeClientEventInputAudioBufferCommit;
template <typename Derived> struct RealtimeClientEventResponseCancelBase;
struct RealtimeClientEventResponseCancel;
template <typename Derived> struct RealtimeClientEventResponseCreateBase;
struct RealtimeClientEventResponseCreate;
template <typename Derived> struct RealtimeResponseCreateParamsBase;
struct RealtimeResponseCreateParams;
template <typename Derived> struct RealtimeClientEventSessionUpdateBase;
struct RealtimeClientEventSessionUpdate;
template <typename Derived> struct RealtimeSessionCreateRequestBase;
struct RealtimeSessionCreateRequest;
template <typename Derived> struct RealtimeResponseBase;
struct RealtimeResponse;
template <typename Derived> struct RealtimeServerEventConversationCreatedBase;
struct RealtimeServerEventConversationCreated;
template <typename Derived> struct RealtimeServerEventConversationItemCreatedBase;
struct RealtimeServerEventConversationItemCreated;
template <typename Derived> struct RealtimeServerEventConversationItemDeletedBase;
struct RealtimeServerEventConversationItemDeleted;
template <typename Derived> struct RealtimeServerEventConversationItemInputAudioTranscriptionCompletedBase;
struct RealtimeServerEventConversationItemInputAudioTranscriptionCompleted;
template <typename Derived> struct RealtimeServerEventConversationItemInputAudioTranscriptionFailedBase;
struct RealtimeServerEventConversationItemInputAudioTranscriptionFailed;
template <typename Derived> struct RealtimeServerEventConversationItemTruncatedBase;
struct RealtimeServerEventConversationItemTruncated;
template <typename Derived> struct RealtimeServerEventErrorBase;
struct RealtimeServerEventError;
template <typename Derived> struct RealtimeServerEventInputAudioBufferClearedBase;
struct RealtimeServerEventInputAudioBufferCleared;
template <typename Derived> struct RealtimeServerEventInputAudioBufferCommittedBase;
struct RealtimeServerEventInputAudioBufferCommitted;
template <typename Derived> struct RealtimeServerEventInputAudioBufferSpeechStartedBase;
struct RealtimeServerEventInputAudioBufferSpeechStarted;
template <typename Derived> struct RealtimeServerEventInputAudioBufferSpeechStoppedBase;
struct RealtimeServerEventInputAudioBufferSpeechStopped;
template <typename Derived> struct RealtimeServerEventRateLimitsUpdatedBase;
struct RealtimeServerEventRateLimitsUpdated;
template <typename Derived> struct RealtimeServerEventResponseAudioDeltaBase;
struct RealtimeServerEventResponseAudioDelta;
template <typename Derived> struct RealtimeServerEventResponseAudioDoneBase;
struct RealtimeServerEventResponseAudioDone;
template <typename Derived> struct RealtimeServerEventResponseAudioTranscriptDeltaBase;
struct RealtimeServerEventResponseAudioTranscriptDelta;
template <typename Derived> struct RealtimeServerEventResponseAudioTranscriptDoneBase;
struct RealtimeServerEventResponseAudioTranscriptDone;
template <typename Derived> struct RealtimeServerEventResponseContentPartAddedBase;
struct RealtimeServerEventResponseContentPartAdded;
template <typename Derived> struct RealtimeServerEventResponseContentPartDoneBase;
struct RealtimeServerEventResponseContentPartDone;
template <typename Derived> struct RealtimeServerEventResponseCreatedBase;
struct RealtimeServerEventResponseCreated;
template <typename Derived> struct RealtimeServerEventResponseDoneBase;
struct RealtimeServerEventResponseDone;
template <typename Derived> struct RealtimeServerEventResponseFunctionCallArgumentsDeltaBase;
struct RealtimeServerEventResponseFunctionCallArgumentsDelta;
template <typename Derived> struct RealtimeServerEventResponseFunctionCallArgumentsDoneBase;
struct RealtimeServerEventResponseFunctionCallArgumentsDone;
template <typename Derived> struct RealtimeServerEventResponseOutputItemAddedBase;
struct RealtimeServerEventResponseOutputItemAdded;
template <typename Derived> struct RealtimeServerEventResponseOutputItemDoneBase;
struct RealtimeServerEventResponseOutputItemDone;
template <typename Derived> struct RealtimeServerEventResponseTextDeltaBase;
struct RealtimeServerEventResponseTextDelta;
template <typename Derived> struct RealtimeServerEventResponseTextDoneBase;
struct RealtimeServerEventResponseTextDone;
template <typename Derived> struct RealtimeServerEventSessionCreatedBase;
struct RealtimeServerEventSessionCreated;
template <typename Derived> struct RealtimeSessionBase;
struct RealtimeSession;
template <typename Derived> struct RealtimeServerEventSessionUpdatedBase;
struct RealtimeServerEventSessionUpdated;
template <typename Derived> struct RealtimeSessionCreateResponseBase;
struct RealtimeSessionCreateResponse;
template <typename Derived> struct ResponseFormatJsonObjectBase;
struct ResponseFormatJsonObject;
template <typename Derived> struct ResponseFormatJsonSchemaBase;
struct ResponseFormatJsonSchema;
template <typename Derived> struct ResponseFormatJsonSchemaSchemaBase;
struct ResponseFormatJsonSchemaSchema;
template <typename Derived> struct ResponseFormatTextBase;
struct ResponseFormatText;
template <typename Derived> struct RunCompletionUsageBase;
struct RunCompletionUsage;
template <typename Derived> struct RunObjectBase;
struct RunObject;
template <typename Derived> struct RunStepCompletionUsageBase;
struct RunStepCompletionUsage;
template <typename Derived> struct RunStepDeltaObjectBase;
struct RunStepDeltaObject;
template <typename Derived> struct RunStepDeltaStepDetailsMessageCreationObjectBase;
struct RunStepDeltaStepDetailsMessageCreationObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsCodeObjectBase;
struct RunStepDeltaStepDetailsToolCallsCodeObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObjectBase;
struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObjectBase;
struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsFileSearchObjectBase;
struct RunStepDeltaStepDetailsToolCallsFileSearchObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsFunctionObjectBase;
struct RunStepDeltaStepDetailsToolCallsFunctionObject;
template <typename Derived> struct RunStepDeltaStepDetailsToolCallsObjectBase;
struct RunStepDeltaStepDetailsToolCallsObject;
template <typename Derived> struct RunStepDetailsMessageCreationObjectBase;
struct RunStepDetailsMessageCreationObject;
template <typename Derived> struct RunStepDetailsToolCallsCodeObjectBase;
struct RunStepDetailsToolCallsCodeObject;
template <typename Derived> struct RunStepDetailsToolCallsCodeOutputImageObjectBase;
struct RunStepDetailsToolCallsCodeOutputImageObject;
template <typename Derived> struct RunStepDetailsToolCallsCodeOutputLogsObjectBase;
struct RunStepDetailsToolCallsCodeOutputLogsObject;
template <typename Derived> struct RunStepDetailsToolCallsFileSearchObjectBase;
struct RunStepDetailsToolCallsFileSearchObject;
template <typename Derived> struct RunStepDetailsToolCallsFileSearchRankingOptionsObjectBase;
struct RunStepDetailsToolCallsFileSearchRankingOptionsObject;
template <typename Derived> struct RunStepDetailsToolCallsFileSearchResultObjectBase;
struct RunStepDetailsToolCallsFileSearchResultObject;
template <typename Derived> struct RunStepDetailsToolCallsFunctionObjectBase;
struct RunStepDetailsToolCallsFunctionObject;
template <typename Derived> struct RunStepDetailsToolCallsObjectBase;
struct RunStepDetailsToolCallsObject;
template <typename Derived> struct RunStepObjectBase;
struct RunStepObject;
template <typename Derived> struct RunStepStreamEventBase;
struct RunStepStreamEvent;
template <typename Derived> struct RunStreamEventBase;
struct RunStreamEvent;
template <typename Derived> struct RunToolCallObjectBase;
struct RunToolCallObject;
template <typename Derived> struct StaticChunkingStrategyBase;
struct StaticChunkingStrategy;
template <typename Derived> struct StaticChunkingStrategyRequestParamBase;
struct StaticChunkingStrategyRequestParam;
template <typename Derived> struct StaticChunkingStrategyResponseParamBase;
struct StaticChunkingStrategyResponseParam;
template <typename Derived> struct SubmitToolOutputsRunRequestBase;
struct SubmitToolOutputsRunRequest;
template <typename Derived> struct ThreadObjectBase;
struct ThreadObject;
template <typename Derived> struct ThreadStreamEventBase;
struct ThreadStreamEvent;
template <typename Derived> struct TranscriptionSegmentBase;
struct TranscriptionSegment;
template <typename Derived> struct TranscriptionWordBase;
struct TranscriptionWord;
template <typename Derived> struct UpdateVectorStoreRequestBase;
struct UpdateVectorStoreRequest;
template <typename Derived> struct UploadBase;
struct Upload;
template <typename Derived> struct UploadPartBase;
struct UploadPart;
template <typename Derived> struct UsageAudioSpeechesResultBase;
struct UsageAudioSpeechesResult;
template <typename Derived> struct UsageAudioTranscriptionsResultBase;
struct UsageAudioTranscriptionsResult;
template <typename Derived> struct UsageCodeInterpreterSessionsResultBase;
struct UsageCodeInterpreterSessionsResult;
template <typename Derived> struct UsageCompletionsResultBase;
struct UsageCompletionsResult;
template <typename Derived> struct UsageEmbeddingsResultBase;
struct UsageEmbeddingsResult;
template <typename Derived> struct UsageImagesResultBase;
struct UsageImagesResult;
template <typename Derived> struct UsageModerationsResultBase;
struct UsageModerationsResult;
template <typename Derived> struct UsageResponseBase;
struct UsageResponse;
template <typename Derived> struct UsageTimeBucketBase;
struct UsageTimeBucket;
template <typename Derived> struct UsageVectorStoresResultBase;
struct UsageVectorStoresResult;
template <typename Derived> struct UserBase;
struct User;
template <typename Derived> struct UserDeleteResponseBase;
struct UserDeleteResponse;
template <typename Derived> struct UserListResponseBase;
struct UserListResponse;
template <typename Derived> struct UserRoleUpdateRequestBase;
struct UserRoleUpdateRequest;
template <typename Derived> struct VectorStoreFileBatchObjectBase;
struct VectorStoreFileBatchObject;
template <typename Derived> struct VectorStoreFileObjectBase;
struct VectorStoreFileObject;
template <typename Derived> struct VectorStoreObjectBase;
struct VectorStoreObject;

// Base definitions (CRTP):
template <typename Derived>
struct AddUploadPartRequestBase {
    UNKNOWN_TYPE data;
};

template <typename Derived>
struct AdminApiKeyBase {
    std::string object;
    std::string id;
    std::string name;
    std::string redacted_value;
    std::string value;
    long long created_at;
    UNKNOWN_TYPE owner;
};

template <typename Derived>
struct ApiKeyListBase {
    std::string object;
    std::vector<AdminApiKey> data;
    bool has_more;
    std::string first_id;
    std::string last_id;
};

template <typename Derived>
struct AssistantsApiResponseFormatOptionBase {
    // no direct properties
};

template <typename Derived>
struct AssistantObjectBase : public AssistantsApiResponseFormatOptionBase<Derived> {
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
};

template <typename Derived>
struct AssistantStreamEventBase {
    // no direct properties
};

template <typename Derived>
struct AssistantToolsCodeBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct AssistantToolsFileSearchBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

template <typename Derived>
struct AssistantToolsFileSearchTypeOnlyBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct FunctionParametersBase {
    // no direct properties
};

template <typename Derived>
struct FunctionObjectBase : public FunctionParametersBase<Derived> {
    std::string description;
    std::string name;
    bool strict;
};

template <typename Derived>
struct AssistantToolsFunctionBase : public FunctionObjectBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct AssistantsApiToolChoiceOptionBase {
    // no direct properties
};

template <typename Derived>
struct AssistantsNamedToolChoiceBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct AudioResponseFormatBase {
    // no direct properties
};

template <typename Derived>
struct AuditLogEventTypeBase {
    // no direct properties
};

template <typename Derived>
struct AuditLogActorUserBase {
    std::string id;
    std::string email;
};

template <typename Derived>
struct AuditLogActorSessionBase : public AuditLogActorUserBase<Derived> {
    std::string ip_address;
};

template <typename Derived>
struct AuditLogActorServiceAccountBase {
    std::string id;
};

template <typename Derived>
struct AuditLogActorApiKeyBase : public AuditLogActorUserBase<Derived>, public AuditLogActorServiceAccountBase<Derived> {
    std::string id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct AuditLogActorBase : public AuditLogActorSessionBase<Derived>, public AuditLogActorApiKeyBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct AuditLogBase : public AuditLogEventTypeBase<Derived>, public AuditLogActorBase<Derived> {
    std::string id;
    int effective_at;
    UNKNOWN_TYPE project;
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

template <typename Derived>
struct AutoChunkingStrategyRequestParamBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct BatchBase {
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

template <typename Derived>
struct BatchRequestInputBase {
    std::string custom_id;
    std::string  // or generate an enum method;
    std::string url;
};

template <typename Derived>
struct BatchRequestOutputBase {
    std::string id;
    std::string custom_id;
    UNKNOWN_TYPE response;
    UNKNOWN_TYPE error;
};

template <typename Derived>
struct CancelUploadRequestBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionFunctionCallOptionBase {
    std::string name;
};

template <typename Derived>
struct ChatCompletionFunctionsBase : public FunctionParametersBase<Derived> {
    std::string description;
    std::string name;
};

template <typename Derived>
struct ChatCompletionMessageToolCallBase {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct ChatCompletionMessageToolCallChunkBase {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct ChatCompletionMessageToolCallsBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionModalitiesBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionNamedToolChoiceBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct ChatCompletionRequestAssistantMessageBase : public ChatCompletionMessageToolCallsBase<Derived> {
    UNKNOWN_TYPE content;
    std::string refusal;
    std::string  // or generate an enum role;
    std::string name;
    UNKNOWN_TYPE audio;
    UNKNOWN_TYPE function_call;
};

template <typename Derived>
struct ChatCompletionRequestAssistantMessageContentPartBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionRequestDeveloperMessageBase {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

template <typename Derived>
struct ChatCompletionRequestFunctionMessageBase {
    std::string  // or generate an enum role;
    std::string content;
    std::string name;
};

template <typename Derived>
struct ChatCompletionRequestMessageBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionRequestMessageContentPartAudioBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE input_audio;
};

template <typename Derived>
struct ChatCompletionRequestMessageContentPartImageBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

template <typename Derived>
struct ChatCompletionRequestMessageContentPartRefusalBase {
    std::string  // or generate an enum type;
    std::string refusal;
};

template <typename Derived>
struct ChatCompletionRequestMessageContentPartTextBase {
    std::string  // or generate an enum type;
    std::string text;
};

template <typename Derived>
struct ChatCompletionRequestSystemMessageBase {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

template <typename Derived>
struct ChatCompletionRequestSystemMessageContentPartBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionRequestToolMessageBase {
    std::string  // or generate an enum role;
    UNKNOWN_TYPE content;
    std::string tool_call_id;
};

template <typename Derived>
struct ChatCompletionRequestToolMessageContentPartBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionRequestUserMessageBase {
    UNKNOWN_TYPE content;
    std::string  // or generate an enum role;
    std::string name;
};

template <typename Derived>
struct ChatCompletionRequestUserMessageContentPartBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionResponseMessageBase : public ChatCompletionMessageToolCallsBase<Derived> {
    std::string content;
    std::string refusal;
    std::string  // or generate an enum role;
    UNKNOWN_TYPE function_call;
    UNKNOWN_TYPE audio;
};

template <typename Derived>
struct ChatCompletionRoleBase {
    // no direct properties
};

template <typename Derived>
struct ChatCompletionStreamOptionsBase {
    bool include_usage;
};

template <typename Derived>
struct ChatCompletionStreamResponseDeltaBase {
    std::string content;
    UNKNOWN_TYPE function_call;
    std::vector<ChatCompletionMessageToolCallChunk> tool_calls;
    std::string  // or generate an enum role;
    std::string refusal;
};

template <typename Derived>
struct ChatCompletionTokenLogprobBase {
    std::string token;
    double logprob;
    std::vector<int> bytes;
    std::vector<UNKNOWN_TYPE> top_logprobs;
};

template <typename Derived>
struct ChatCompletionToolBase : public FunctionObjectBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct ChatCompletionToolChoiceOptionBase {
    // no direct properties
};

template <typename Derived>
struct ChunkingStrategyRequestParamBase {
    // no direct properties
};

template <typename Derived>
struct CompleteUploadRequestBase {
    std::vector<std::string> part_ids;
    std::string md5;
};

template <typename Derived>
struct CompletionUsageBase {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
    UNKNOWN_TYPE completion_tokens_details;
    UNKNOWN_TYPE prompt_tokens_details;
};

template <typename Derived>
struct CostsResultBase {
    std::string  // or generate an enum object;
    UNKNOWN_TYPE amount;
    std::string line_item;
    std::string project_id;
};

template <typename Derived>
struct CreateAssistantRequestBase : public AssistantsApiResponseFormatOptionBase<Derived> {
    UNKNOWN_TYPE model;
    std::string name;
    std::string description;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
};

template <typename Derived>
struct CreateChatCompletionFunctionResponseBase : public CompletionUsageBase<Derived> {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct CreateChatCompletionImageResponseBase {
    // no direct properties
};

template <typename Derived>
struct ParallelToolCallsBase {
    // no direct properties
};

template <typename Derived>
struct CreateChatCompletionRequestBase : public ChatCompletionModalitiesBase<Derived>, public ChatCompletionStreamOptionsBase<Derived>, public ChatCompletionToolChoiceOptionBase<Derived>, public ParallelToolCallsBase<Derived> {
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
    UNKNOWN_TYPE prediction;
    UNKNOWN_TYPE audio;
    double presence_penalty;
    UNKNOWN_TYPE response_format;
    int seed;
    std::string  // or generate an enum service_tier;
    UNKNOWN_TYPE stop;
    bool stream;
    double temperature;
    double top_p;
    std::vector<ChatCompletionTool> tools;
    std::string user;
    UNKNOWN_TYPE function_call;
    std::vector<ChatCompletionFunctions> functions;
};

template <typename Derived>
struct CreateChatCompletionResponseBase : public CompletionUsageBase<Derived> {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string  // or generate an enum service_tier;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct CreateChatCompletionStreamResponseBase {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string  // or generate an enum service_tier;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE usage;
};

template <typename Derived>
struct CreateCompletionRequestBase : public ChatCompletionStreamOptionsBase<Derived> {
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
    std::string suffix;
    double temperature;
    double top_p;
    std::string user;
};

template <typename Derived>
struct CreateCompletionResponseBase : public CompletionUsageBase<Derived> {
    std::string id;
    std::vector<UNKNOWN_TYPE> choices;
    int created;
    std::string model;
    std::string system_fingerprint;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct CreateEmbeddingRequestBase {
    UNKNOWN_TYPE input;
    UNKNOWN_TYPE model;
    std::string  // or generate an enum encoding_format;
    int dimensions;
    std::string user;
};

template <typename Derived>
struct CreateEmbeddingResponseBase {
    std::vector<Embedding> data;
    std::string model;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE usage;
};

template <typename Derived>
struct CreateFileRequestBase {
    UNKNOWN_TYPE file;
    std::string  // or generate an enum purpose;
};

template <typename Derived>
struct FineTuneSupervisedMethodBase {
    UNKNOWN_TYPE hyperparameters;
};

template <typename Derived>
struct FineTuneDPOMethodBase {
    UNKNOWN_TYPE hyperparameters;
};

template <typename Derived>
struct FineTuneMethodBase : public FineTuneSupervisedMethodBase<Derived>, public FineTuneDPOMethodBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct CreateFineTuningJobRequestBase : public FineTuneMethodBase<Derived> {
    UNKNOWN_TYPE model;
    std::string training_file;
    UNKNOWN_TYPE hyperparameters;
    std::string suffix;
    std::string validation_file;
    std::vector<UNKNOWN_TYPE> integrations;
    int seed;
};

template <typename Derived>
struct CreateImageEditRequestBase {
    UNKNOWN_TYPE image;
    std::string prompt;
    UNKNOWN_TYPE mask;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum size;
    std::string  // or generate an enum response_format;
    std::string user;
};

template <typename Derived>
struct CreateImageRequestBase {
    std::string prompt;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum quality;
    std::string  // or generate an enum response_format;
    std::string  // or generate an enum size;
    std::string  // or generate an enum style;
    std::string user;
};

template <typename Derived>
struct CreateImageVariationRequestBase {
    UNKNOWN_TYPE image;
    UNKNOWN_TYPE model;
    int n;
    std::string  // or generate an enum response_format;
    std::string  // or generate an enum size;
    std::string user;
};

template <typename Derived>
struct CreateMessageRequestBase {
    std::string  // or generate an enum role;
    UNKNOWN_TYPE content;
    std::vector<UNKNOWN_TYPE> attachments;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct CreateModerationRequestBase {
    UNKNOWN_TYPE input;
    UNKNOWN_TYPE model;
};

template <typename Derived>
struct CreateModerationResponseBase {
    std::string id;
    std::string model;
    std::vector<UNKNOWN_TYPE> results;
};

template <typename Derived>
struct TruncationObjectBase {
    std::string  // or generate an enum type;
    int last_messages;
};

template <typename Derived>
struct CreateRunRequestBase : public TruncationObjectBase<Derived>, public AssistantsApiToolChoiceOptionBase<Derived>, public ParallelToolCallsBase<Derived>, public AssistantsApiResponseFormatOptionBase<Derived> {
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
};

template <typename Derived>
struct CreateSpeechRequestBase {
    UNKNOWN_TYPE model;
    std::string input;
    std::string  // or generate an enum voice;
    std::string  // or generate an enum response_format;
    double speed;
};

template <typename Derived>
struct CreateThreadRequestBase {
    std::vector<CreateMessageRequest> messages;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct CreateThreadAndRunRequestBase : public CreateThreadRequestBase<Derived>, public TruncationObjectBase<Derived>, public AssistantsApiToolChoiceOptionBase<Derived>, public ParallelToolCallsBase<Derived>, public AssistantsApiResponseFormatOptionBase<Derived> {
    std::string assistant_id;
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
};

template <typename Derived>
struct CreateTranscriptionRequestBase : public AudioResponseFormatBase<Derived> {
    UNKNOWN_TYPE file;
    UNKNOWN_TYPE model;
    std::string language;
    std::string prompt;
    double temperature;
    std::vector<std::string  // or generate an enum> timestamp_granularities[];
};

template <typename Derived>
struct CreateTranscriptionResponseJsonBase {
    std::string text;
};

template <typename Derived>
struct CreateTranscriptionResponseVerboseJsonBase {
    std::string language;
    std::string duration;
    std::string text;
    std::vector<TranscriptionWord> words;
    std::vector<TranscriptionSegment> segments;
};

template <typename Derived>
struct CreateTranslationRequestBase : public AudioResponseFormatBase<Derived> {
    UNKNOWN_TYPE file;
    UNKNOWN_TYPE model;
    std::string prompt;
    double temperature;
};

template <typename Derived>
struct CreateTranslationResponseJsonBase {
    std::string text;
};

template <typename Derived>
struct CreateTranslationResponseVerboseJsonBase {
    std::string language;
    std::string duration;
    std::string text;
    std::vector<TranscriptionSegment> segments;
};

template <typename Derived>
struct CreateUploadRequestBase {
    std::string filename;
    std::string  // or generate an enum purpose;
    int bytes;
    std::string mime_type;
};

template <typename Derived>
struct CreateVectorStoreFileBatchRequestBase : public ChunkingStrategyRequestParamBase<Derived> {
    std::vector<std::string> file_ids;
};

template <typename Derived>
struct CreateVectorStoreFileRequestBase : public ChunkingStrategyRequestParamBase<Derived> {
    std::string file_id;
};

template <typename Derived>
struct VectorStoreExpirationAfterBase {
    std::string  // or generate an enum anchor;
    int days;
};

template <typename Derived>
struct CreateVectorStoreRequestBase : public VectorStoreExpirationAfterBase<Derived> {
    std::vector<std::string> file_ids;
    std::string name;
    UNKNOWN_TYPE chunking_strategy;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct DefaultProjectErrorResponseBase {
    int code;
    std::string message;
};

template <typename Derived>
struct DeleteAssistantResponseBase {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct DeleteFileResponseBase {
    std::string id;
    std::string  // or generate an enum object;
    bool deleted;
};

template <typename Derived>
struct DeleteMessageResponseBase {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct DeleteModelResponseBase {
    std::string id;
    bool deleted;
    std::string object;
};

template <typename Derived>
struct DeleteThreadResponseBase {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct DeleteVectorStoreFileResponseBase {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct DeleteVectorStoreResponseBase {
    std::string id;
    bool deleted;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct DoneEventBase {
    std::string  // or generate an enum event;
    std::string  // or generate an enum data;
};

template <typename Derived>
struct EmbeddingBase {
    int index;
    std::vector<double> embedding;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct ErrorBase {
    std::string code;
    std::string message;
    std::string param;
    std::string type;
};

template <typename Derived>
struct ErrorEventBase : public ErrorBase<Derived> {
    std::string  // or generate an enum event;
};

template <typename Derived>
struct ErrorResponseBase : public ErrorBase<Derived> {
    // no direct properties
};

template <typename Derived>
struct FileSearchRankingOptionsBase {
    std::string  // or generate an enum ranker;
    double score_threshold;
};

template <typename Derived>
struct FineTuneChatCompletionRequestAssistantMessageBase {
    // no direct properties
};

template <typename Derived>
struct FineTuneChatRequestInputBase : public ParallelToolCallsBase<Derived> {
    std::vector<UNKNOWN_TYPE> messages;
    std::vector<ChatCompletionTool> tools;
    std::vector<ChatCompletionFunctions> functions;
};

template <typename Derived>
struct FineTuneCompletionRequestInputBase {
    std::string prompt;
    std::string completion;
};

template <typename Derived>
struct FineTunePreferenceRequestInputBase {
    UNKNOWN_TYPE input;
    std::vector<UNKNOWN_TYPE> preferred_completion;
    std::vector<UNKNOWN_TYPE> non_preferred_completion;
};

template <typename Derived>
struct FineTuningIntegrationBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE wandb;
};

template <typename Derived>
struct FineTuningJobBase : public FineTuneMethodBase<Derived> {
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
};

template <typename Derived>
struct FineTuningJobCheckpointBase {
    std::string id;
    int created_at;
    std::string fine_tuned_model_checkpoint;
    int step_number;
    UNKNOWN_TYPE metrics;
    std::string fine_tuning_job_id;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct FineTuningJobEventBase {
    std::string  // or generate an enum object;
    std::string id;
    int created_at;
    std::string  // or generate an enum level;
    std::string message;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE data;
};

template <typename Derived>
struct ImageBase {
    std::string b64_json;
    std::string url;
    std::string revised_prompt;
};

template <typename Derived>
struct ImagesResponseBase {
    int created;
    std::vector<Image> data;
};

template <typename Derived>
struct InviteBase {
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

template <typename Derived>
struct InviteDeleteResponseBase {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

template <typename Derived>
struct InviteListResponseBase {
    std::string  // or generate an enum object;
    std::vector<Invite> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct InviteRequestBase {
    std::string email;
    std::string  // or generate an enum role;
    std::vector<UNKNOWN_TYPE> projects;
};

template <typename Derived>
struct ListAssistantsResponseBase {
    std::string object;
    std::vector<AssistantObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListAuditLogsResponseBase {
    std::string  // or generate an enum object;
    std::vector<AuditLog> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListBatchesResponseBase {
    std::vector<Batch> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct ListFilesResponseBase {
    std::string object;
    std::vector<OpenAIFile> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListFineTuningJobCheckpointsResponseBase {
    std::vector<FineTuningJobCheckpoint> data;
    std::string  // or generate an enum object;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListFineTuningJobEventsResponseBase {
    std::vector<FineTuningJobEvent> data;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct ListMessagesResponseBase {
    std::string object;
    std::vector<MessageObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListModelsResponseBase {
    std::string  // or generate an enum object;
    std::vector<Model> data;
};

template <typename Derived>
struct ListPaginatedFineTuningJobsResponseBase {
    std::vector<FineTuningJob> data;
    bool has_more;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct ListRunStepsResponseBase {
    std::string object;
    std::vector<RunStepObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListRunsResponseBase {
    std::string object;
    std::vector<RunObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListThreadsResponseBase {
    std::string object;
    std::vector<ThreadObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListVectorStoreFilesResponseBase {
    std::string object;
    std::vector<VectorStoreFileObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ListVectorStoresResponseBase {
    std::string object;
    std::vector<VectorStoreObject> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct MessageContentImageFileObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_file;
};

template <typename Derived>
struct MessageContentImageUrlObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

template <typename Derived>
struct MessageContentRefusalObjectBase {
    std::string  // or generate an enum type;
    std::string refusal;
};

template <typename Derived>
struct MessageContentTextAnnotationsFileCitationObjectBase {
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_citation;
    int start_index;
    int end_index;
};

template <typename Derived>
struct MessageContentTextAnnotationsFilePathObjectBase {
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_path;
    int start_index;
    int end_index;
};

template <typename Derived>
struct MessageContentTextObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE text;
};

template <typename Derived>
struct MessageDeltaContentImageFileObjectBase {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_file;
};

template <typename Derived>
struct MessageDeltaContentImageUrlObjectBase {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image_url;
};

template <typename Derived>
struct MessageDeltaContentRefusalObjectBase {
    int index;
    std::string  // or generate an enum type;
    std::string refusal;
};

template <typename Derived>
struct MessageDeltaContentTextAnnotationsFileCitationObjectBase {
    int index;
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_citation;
    int start_index;
    int end_index;
};

template <typename Derived>
struct MessageDeltaContentTextAnnotationsFilePathObjectBase {
    int index;
    std::string  // or generate an enum type;
    std::string text;
    UNKNOWN_TYPE file_path;
    int start_index;
    int end_index;
};

template <typename Derived>
struct MessageDeltaContentTextObjectBase {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE text;
};

template <typename Derived>
struct MessageDeltaObjectBase {
    std::string id;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE delta;
};

template <typename Derived>
struct MessageObjectBase {
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

template <typename Derived>
struct MessageRequestContentTextObjectBase {
    std::string  // or generate an enum type;
    std::string text;
};

template <typename Derived>
struct MessageStreamEventBase {
    // no direct properties
};

template <typename Derived>
struct ModelBase {
    std::string id;
    int created;
    std::string  // or generate an enum object;
    std::string owned_by;
};

template <typename Derived>
struct ModifyAssistantRequestBase : public AssistantsApiResponseFormatOptionBase<Derived> {
    UNKNOWN_TYPE model;
    std::string name;
    std::string description;
    std::string instructions;
    std::vector<UNKNOWN_TYPE> tools;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
    double temperature;
    double top_p;
};

template <typename Derived>
struct ModifyMessageRequestBase {
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct ModifyRunRequestBase {
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct ModifyThreadRequestBase {
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct OpenAIFileBase {
    std::string id;
    int bytes;
    int created_at;
    std::string filename;
    std::string  // or generate an enum object;
    std::string  // or generate an enum purpose;
    std::string  // or generate an enum status;
    std::string status_details;
};

template <typename Derived>
struct OtherChunkingStrategyResponseParamBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct PredictionContentBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE content;
};

template <typename Derived>
struct ProjectBase {
    std::string id;
    std::string  // or generate an enum object;
    std::string name;
    int created_at;
    int archived_at;
    std::string  // or generate an enum status;
};

template <typename Derived>
struct ProjectApiKeyBase {
    std::string  // or generate an enum object;
    std::string redacted_value;
    std::string name;
    int created_at;
    std::string id;
    UNKNOWN_TYPE owner;
};

template <typename Derived>
struct ProjectApiKeyDeleteResponseBase {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

template <typename Derived>
struct ProjectApiKeyListResponseBase {
    std::string  // or generate an enum object;
    std::vector<ProjectApiKey> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ProjectCreateRequestBase {
    std::string name;
};

template <typename Derived>
struct ProjectListResponseBase {
    std::string  // or generate an enum object;
    std::vector<Project> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ProjectRateLimitBase {
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

template <typename Derived>
struct ProjectRateLimitListResponseBase {
    std::string  // or generate an enum object;
    std::vector<ProjectRateLimit> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ProjectRateLimitUpdateRequestBase {
    int max_requests_per_1_minute;
    int max_tokens_per_1_minute;
    int max_images_per_1_minute;
    int max_audio_megabytes_per_1_minute;
    int max_requests_per_1_day;
    int batch_1_day_max_input_tokens;
};

template <typename Derived>
struct ProjectServiceAccountBase {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string  // or generate an enum role;
    int created_at;
};

template <typename Derived>
struct ProjectServiceAccountApiKeyBase {
    std::string  // or generate an enum object;
    std::string value;
    std::string name;
    int created_at;
    std::string id;
};

template <typename Derived>
struct ProjectServiceAccountCreateRequestBase {
    std::string name;
};

template <typename Derived>
struct ProjectServiceAccountCreateResponseBase : public ProjectServiceAccountApiKeyBase<Derived> {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string  // or generate an enum role;
    int created_at;
};

template <typename Derived>
struct ProjectServiceAccountDeleteResponseBase {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

template <typename Derived>
struct ProjectServiceAccountListResponseBase {
    std::string  // or generate an enum object;
    std::vector<ProjectServiceAccount> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ProjectUpdateRequestBase {
    std::string name;
};

template <typename Derived>
struct ProjectUserBase {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string email;
    std::string  // or generate an enum role;
    int added_at;
};

template <typename Derived>
struct ProjectUserCreateRequestBase {
    std::string user_id;
    std::string  // or generate an enum role;
};

template <typename Derived>
struct ProjectUserDeleteResponseBase {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

template <typename Derived>
struct ProjectUserListResponseBase {
    std::string object;
    std::vector<ProjectUser> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct ProjectUserUpdateRequestBase {
    std::string  // or generate an enum role;
};

template <typename Derived>
struct RealtimeConversationItemBase {
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

template <typename Derived>
struct RealtimeClientEventConversationItemCreateBase : public RealtimeConversationItemBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
};

template <typename Derived>
struct RealtimeClientEventConversationItemDeleteBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
};

template <typename Derived>
struct RealtimeClientEventConversationItemTruncateBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    int audio_end_ms;
};

template <typename Derived>
struct RealtimeClientEventInputAudioBufferAppendBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string audio;
};

template <typename Derived>
struct RealtimeClientEventInputAudioBufferClearBase {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeClientEventInputAudioBufferCommitBase {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeClientEventResponseCancelBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
};

template <typename Derived>
struct RealtimeResponseCreateParamsBase {
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

template <typename Derived>
struct RealtimeClientEventResponseCreateBase : public RealtimeResponseCreateParamsBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeSessionCreateRequestBase {
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

template <typename Derived>
struct RealtimeClientEventSessionUpdateBase : public RealtimeSessionCreateRequestBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeResponseBase {
    std::string id;
    std::string  // or generate an enum object;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE status_details;
    std::vector<RealtimeConversationItem> output;
    UNKNOWN_TYPE metadata;
    UNKNOWN_TYPE usage;
};

template <typename Derived>
struct RealtimeServerEventConversationCreatedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE conversation;
};

template <typename Derived>
struct RealtimeServerEventConversationItemCreatedBase : public RealtimeConversationItemBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
};

template <typename Derived>
struct RealtimeServerEventConversationItemDeletedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
};

template <typename Derived>
struct RealtimeServerEventConversationItemInputAudioTranscriptionCompletedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    std::string transcript;
};

template <typename Derived>
struct RealtimeServerEventConversationItemInputAudioTranscriptionFailedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    UNKNOWN_TYPE error;
};

template <typename Derived>
struct RealtimeServerEventConversationItemTruncatedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string item_id;
    int content_index;
    int audio_end_ms;
};

template <typename Derived>
struct RealtimeServerEventErrorBase {
    std::string event_id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE error;
};

template <typename Derived>
struct RealtimeServerEventInputAudioBufferClearedBase {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeServerEventInputAudioBufferCommittedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string previous_item_id;
    std::string item_id;
};

template <typename Derived>
struct RealtimeServerEventInputAudioBufferSpeechStartedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    int audio_start_ms;
    std::string item_id;
};

template <typename Derived>
struct RealtimeServerEventInputAudioBufferSpeechStoppedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    int audio_end_ms;
    std::string item_id;
};

template <typename Derived>
struct RealtimeServerEventRateLimitsUpdatedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> rate_limits;
};

template <typename Derived>
struct RealtimeServerEventResponseAudioDeltaBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

template <typename Derived>
struct RealtimeServerEventResponseAudioDoneBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
};

template <typename Derived>
struct RealtimeServerEventResponseAudioTranscriptDeltaBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

template <typename Derived>
struct RealtimeServerEventResponseAudioTranscriptDoneBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string transcript;
};

template <typename Derived>
struct RealtimeServerEventResponseContentPartAddedBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    UNKNOWN_TYPE part;
};

template <typename Derived>
struct RealtimeServerEventResponseContentPartDoneBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    UNKNOWN_TYPE part;
};

template <typename Derived>
struct RealtimeServerEventResponseCreatedBase : public RealtimeResponseBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeServerEventResponseDoneBase : public RealtimeResponseBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeServerEventResponseFunctionCallArgumentsDeltaBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    std::string call_id;
    std::string delta;
};

template <typename Derived>
struct RealtimeServerEventResponseFunctionCallArgumentsDoneBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    std::string call_id;
    std::string arguments;
};

template <typename Derived>
struct RealtimeServerEventResponseOutputItemAddedBase : public RealtimeConversationItemBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    int output_index;
};

template <typename Derived>
struct RealtimeServerEventResponseOutputItemDoneBase : public RealtimeConversationItemBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    int output_index;
};

template <typename Derived>
struct RealtimeServerEventResponseTextDeltaBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string delta;
};

template <typename Derived>
struct RealtimeServerEventResponseTextDoneBase {
    std::string event_id;
    std::string  // or generate an enum type;
    std::string response_id;
    std::string item_id;
    int output_index;
    int content_index;
    std::string text;
};

template <typename Derived>
struct RealtimeSessionBase {
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

template <typename Derived>
struct RealtimeServerEventSessionCreatedBase : public RealtimeSessionBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeServerEventSessionUpdatedBase : public RealtimeSessionBase<Derived> {
    std::string event_id;
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RealtimeSessionCreateResponseBase {
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

template <typename Derived>
struct ResponseFormatJsonObjectBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct ResponseFormatJsonSchemaBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE json_schema;
};

template <typename Derived>
struct ResponseFormatJsonSchemaSchemaBase {
    // no direct properties
};

template <typename Derived>
struct ResponseFormatTextBase {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct RunCompletionUsageBase {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
};

template <typename Derived>
struct RunObjectBase : public RunCompletionUsageBase<Derived>, public TruncationObjectBase<Derived>, public AssistantsApiToolChoiceOptionBase<Derived>, public ParallelToolCallsBase<Derived>, public AssistantsApiResponseFormatOptionBase<Derived> {
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
    double temperature;
    double top_p;
    int max_prompt_tokens;
    int max_completion_tokens;
};

template <typename Derived>
struct RunStepCompletionUsageBase {
    int completion_tokens;
    int prompt_tokens;
    int total_tokens;
};

template <typename Derived>
struct RunStepDeltaObjectBase {
    std::string id;
    std::string  // or generate an enum object;
    UNKNOWN_TYPE delta;
};

template <typename Derived>
struct RunStepDeltaStepDetailsMessageCreationObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE message_creation;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsCodeObjectBase {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE code_interpreter;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObjectBase {
    int index;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObjectBase {
    int index;
    std::string  // or generate an enum type;
    std::string logs;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsFileSearchObjectBase {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsFunctionObjectBase {
    int index;
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct RunStepDeltaStepDetailsToolCallsObjectBase {
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> tool_calls;
};

template <typename Derived>
struct RunStepDetailsMessageCreationObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE message_creation;
};

template <typename Derived>
struct RunStepDetailsToolCallsCodeObjectBase {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE code_interpreter;
};

template <typename Derived>
struct RunStepDetailsToolCallsCodeOutputImageObjectBase {
    std::string  // or generate an enum type;
    UNKNOWN_TYPE image;
};

template <typename Derived>
struct RunStepDetailsToolCallsCodeOutputLogsObjectBase {
    std::string  // or generate an enum type;
    std::string logs;
};

template <typename Derived>
struct RunStepDetailsToolCallsFileSearchObjectBase {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE file_search;
};

template <typename Derived>
struct RunStepDetailsToolCallsFileSearchRankingOptionsObjectBase {
    std::string  // or generate an enum ranker;
    double score_threshold;
};

template <typename Derived>
struct RunStepDetailsToolCallsFileSearchResultObjectBase {
    std::string file_id;
    std::string file_name;
    double score;
    std::vector<UNKNOWN_TYPE> content;
};

template <typename Derived>
struct RunStepDetailsToolCallsFunctionObjectBase {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct RunStepDetailsToolCallsObjectBase {
    std::string  // or generate an enum type;
    std::vector<UNKNOWN_TYPE> tool_calls;
};

template <typename Derived>
struct RunStepObjectBase : public RunStepCompletionUsageBase<Derived> {
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
};

template <typename Derived>
struct RunStepStreamEventBase {
    // no direct properties
};

template <typename Derived>
struct RunStreamEventBase {
    // no direct properties
};

template <typename Derived>
struct RunToolCallObjectBase {
    std::string id;
    std::string  // or generate an enum type;
    UNKNOWN_TYPE function;
};

template <typename Derived>
struct StaticChunkingStrategyBase {
    int max_chunk_size_tokens;
    int chunk_overlap_tokens;
};

template <typename Derived>
struct StaticChunkingStrategyRequestParamBase : public StaticChunkingStrategyBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct StaticChunkingStrategyResponseParamBase : public StaticChunkingStrategyBase<Derived> {
    std::string  // or generate an enum type;
};

template <typename Derived>
struct SubmitToolOutputsRunRequestBase {
    std::vector<UNKNOWN_TYPE> tool_outputs;
    bool stream;
};

template <typename Derived>
struct ThreadObjectBase {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    UNKNOWN_TYPE tool_resources;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct ThreadStreamEventBase {
    // no direct properties
};

template <typename Derived>
struct TranscriptionSegmentBase {
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

template <typename Derived>
struct TranscriptionWordBase {
    std::string word;
    UNKNOWN_TYPE start;
    UNKNOWN_TYPE end;
};

template <typename Derived>
struct UpdateVectorStoreRequestBase : public VectorStoreExpirationAfterBase<Derived> {
    std::string name;
    UNKNOWN_TYPE metadata;
};

template <typename Derived>
struct UploadBase : public OpenAIFileBase<Derived> {
    std::string id;
    int created_at;
    std::string filename;
    int bytes;
    std::string purpose;
    std::string  // or generate an enum status;
    int expires_at;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct UploadPartBase {
    std::string id;
    int created_at;
    std::string upload_id;
    std::string  // or generate an enum object;
};

template <typename Derived>
struct UsageAudioSpeechesResultBase {
    std::string  // or generate an enum object;
    int characters;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

template <typename Derived>
struct UsageAudioTranscriptionsResultBase {
    std::string  // or generate an enum object;
    int seconds;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

template <typename Derived>
struct UsageCodeInterpreterSessionsResultBase {
    std::string  // or generate an enum object;
    int sessions;
    std::string project_id;
};

template <typename Derived>
struct UsageCompletionsResultBase {
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

template <typename Derived>
struct UsageEmbeddingsResultBase {
    std::string  // or generate an enum object;
    int input_tokens;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

template <typename Derived>
struct UsageImagesResultBase {
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

template <typename Derived>
struct UsageModerationsResultBase {
    std::string  // or generate an enum object;
    int input_tokens;
    int num_model_requests;
    std::string project_id;
    std::string user_id;
    std::string api_key_id;
    std::string model;
};

template <typename Derived>
struct UsageResponseBase {
    std::string  // or generate an enum object;
    std::vector<UsageTimeBucket> data;
    bool has_more;
    std::string next_page;
};

template <typename Derived>
struct UsageTimeBucketBase {
    std::string  // or generate an enum object;
    int start_time;
    int end_time;
    std::vector<UNKNOWN_TYPE> result;
};

template <typename Derived>
struct UsageVectorStoresResultBase {
    std::string  // or generate an enum object;
    int usage_bytes;
    std::string project_id;
};

template <typename Derived>
struct UserBase {
    std::string  // or generate an enum object;
    std::string id;
    std::string name;
    std::string email;
    std::string  // or generate an enum role;
    int added_at;
};

template <typename Derived>
struct UserDeleteResponseBase {
    std::string  // or generate an enum object;
    std::string id;
    bool deleted;
};

template <typename Derived>
struct UserListResponseBase {
    std::string  // or generate an enum object;
    std::vector<User> data;
    std::string first_id;
    std::string last_id;
    bool has_more;
};

template <typename Derived>
struct UserRoleUpdateRequestBase {
    std::string  // or generate an enum role;
};

template <typename Derived>
struct VectorStoreFileBatchObjectBase {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string vector_store_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE file_counts;
};

template <typename Derived>
struct VectorStoreFileObjectBase {
    std::string id;
    std::string  // or generate an enum object;
    int usage_bytes;
    int created_at;
    std::string vector_store_id;
    std::string  // or generate an enum status;
    UNKNOWN_TYPE last_error;
    UNKNOWN_TYPE chunking_strategy;
};

template <typename Derived>
struct VectorStoreObjectBase : public VectorStoreExpirationAfterBase<Derived> {
    std::string id;
    std::string  // or generate an enum object;
    int created_at;
    std::string name;
    int usage_bytes;
    UNKNOWN_TYPE file_counts;
    std::string  // or generate an enum status;
    int expires_at;
    int last_active_at;
    UNKNOWN_TYPE metadata;
};


// Derived definitions:
struct AddUploadPartRequest : public AddUploadPartRequestBase<AddUploadPartRequest> {
    // specialized methods if needed
};

struct AdminApiKey : public AdminApiKeyBase<AdminApiKey> {
    // specialized methods if needed
};

struct ApiKeyList : public ApiKeyListBase<ApiKeyList> {
    // specialized methods if needed
};

struct AssistantsApiResponseFormatOption : public AssistantsApiResponseFormatOptionBase<AssistantsApiResponseFormatOption> {
    // specialized methods if needed
};

struct AssistantObject : public AssistantObjectBase<AssistantObject> {
    // specialized methods if needed
};

struct AssistantStreamEvent : public AssistantStreamEventBase<AssistantStreamEvent> {
    // specialized methods if needed
};

struct AssistantToolsCode : public AssistantToolsCodeBase<AssistantToolsCode> {
    // specialized methods if needed
};

struct AssistantToolsFileSearch : public AssistantToolsFileSearchBase<AssistantToolsFileSearch> {
    // specialized methods if needed
};

struct AssistantToolsFileSearchTypeOnly : public AssistantToolsFileSearchTypeOnlyBase<AssistantToolsFileSearchTypeOnly> {
    // specialized methods if needed
};

struct FunctionParameters : public FunctionParametersBase<FunctionParameters> {
    // specialized methods if needed
};

struct FunctionObject : public FunctionObjectBase<FunctionObject> {
    // specialized methods if needed
};

struct AssistantToolsFunction : public AssistantToolsFunctionBase<AssistantToolsFunction> {
    // specialized methods if needed
};

struct AssistantsApiToolChoiceOption : public AssistantsApiToolChoiceOptionBase<AssistantsApiToolChoiceOption> {
    // specialized methods if needed
};

struct AssistantsNamedToolChoice : public AssistantsNamedToolChoiceBase<AssistantsNamedToolChoice> {
    // specialized methods if needed
};

struct AudioResponseFormat : public AudioResponseFormatBase<AudioResponseFormat> {
    // specialized methods if needed
};

struct AuditLogEventType : public AuditLogEventTypeBase<AuditLogEventType> {
    // specialized methods if needed
};

struct AuditLogActorUser : public AuditLogActorUserBase<AuditLogActorUser> {
    // specialized methods if needed
};

struct AuditLogActorSession : public AuditLogActorSessionBase<AuditLogActorSession> {
    // specialized methods if needed
};

struct AuditLogActorServiceAccount : public AuditLogActorServiceAccountBase<AuditLogActorServiceAccount> {
    // specialized methods if needed
};

struct AuditLogActorApiKey : public AuditLogActorApiKeyBase<AuditLogActorApiKey> {
    // specialized methods if needed
};

struct AuditLogActor : public AuditLogActorBase<AuditLogActor> {
    // specialized methods if needed
};

struct AuditLog : public AuditLogBase<AuditLog> {
    // specialized methods if needed
};

struct AutoChunkingStrategyRequestParam : public AutoChunkingStrategyRequestParamBase<AutoChunkingStrategyRequestParam> {
    // specialized methods if needed
};

struct Batch : public BatchBase<Batch> {
    // specialized methods if needed
};

struct BatchRequestInput : public BatchRequestInputBase<BatchRequestInput> {
    // specialized methods if needed
};

struct BatchRequestOutput : public BatchRequestOutputBase<BatchRequestOutput> {
    // specialized methods if needed
};

struct CancelUploadRequest : public CancelUploadRequestBase<CancelUploadRequest> {
    // specialized methods if needed
};

struct ChatCompletionFunctionCallOption : public ChatCompletionFunctionCallOptionBase<ChatCompletionFunctionCallOption> {
    // specialized methods if needed
};

struct ChatCompletionFunctions : public ChatCompletionFunctionsBase<ChatCompletionFunctions> {
    // specialized methods if needed
};

struct ChatCompletionMessageToolCall : public ChatCompletionMessageToolCallBase<ChatCompletionMessageToolCall> {
    // specialized methods if needed
};

struct ChatCompletionMessageToolCallChunk : public ChatCompletionMessageToolCallChunkBase<ChatCompletionMessageToolCallChunk> {
    // specialized methods if needed
};

struct ChatCompletionMessageToolCalls : public ChatCompletionMessageToolCallsBase<ChatCompletionMessageToolCalls> {
    // specialized methods if needed
};

struct ChatCompletionModalities : public ChatCompletionModalitiesBase<ChatCompletionModalities> {
    // specialized methods if needed
};

struct ChatCompletionNamedToolChoice : public ChatCompletionNamedToolChoiceBase<ChatCompletionNamedToolChoice> {
    // specialized methods if needed
};

struct ChatCompletionRequestAssistantMessage : public ChatCompletionRequestAssistantMessageBase<ChatCompletionRequestAssistantMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestAssistantMessageContentPart : public ChatCompletionRequestAssistantMessageContentPartBase<ChatCompletionRequestAssistantMessageContentPart> {
    // specialized methods if needed
};

struct ChatCompletionRequestDeveloperMessage : public ChatCompletionRequestDeveloperMessageBase<ChatCompletionRequestDeveloperMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestFunctionMessage : public ChatCompletionRequestFunctionMessageBase<ChatCompletionRequestFunctionMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestMessage : public ChatCompletionRequestMessageBase<ChatCompletionRequestMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestMessageContentPartAudio : public ChatCompletionRequestMessageContentPartAudioBase<ChatCompletionRequestMessageContentPartAudio> {
    // specialized methods if needed
};

struct ChatCompletionRequestMessageContentPartImage : public ChatCompletionRequestMessageContentPartImageBase<ChatCompletionRequestMessageContentPartImage> {
    // specialized methods if needed
};

struct ChatCompletionRequestMessageContentPartRefusal : public ChatCompletionRequestMessageContentPartRefusalBase<ChatCompletionRequestMessageContentPartRefusal> {
    // specialized methods if needed
};

struct ChatCompletionRequestMessageContentPartText : public ChatCompletionRequestMessageContentPartTextBase<ChatCompletionRequestMessageContentPartText> {
    // specialized methods if needed
};

struct ChatCompletionRequestSystemMessage : public ChatCompletionRequestSystemMessageBase<ChatCompletionRequestSystemMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestSystemMessageContentPart : public ChatCompletionRequestSystemMessageContentPartBase<ChatCompletionRequestSystemMessageContentPart> {
    // specialized methods if needed
};

struct ChatCompletionRequestToolMessage : public ChatCompletionRequestToolMessageBase<ChatCompletionRequestToolMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestToolMessageContentPart : public ChatCompletionRequestToolMessageContentPartBase<ChatCompletionRequestToolMessageContentPart> {
    // specialized methods if needed
};

struct ChatCompletionRequestUserMessage : public ChatCompletionRequestUserMessageBase<ChatCompletionRequestUserMessage> {
    // specialized methods if needed
};

struct ChatCompletionRequestUserMessageContentPart : public ChatCompletionRequestUserMessageContentPartBase<ChatCompletionRequestUserMessageContentPart> {
    // specialized methods if needed
};

struct ChatCompletionResponseMessage : public ChatCompletionResponseMessageBase<ChatCompletionResponseMessage> {
    // specialized methods if needed
};

struct ChatCompletionRole : public ChatCompletionRoleBase<ChatCompletionRole> {
    // specialized methods if needed
};

struct ChatCompletionStreamOptions : public ChatCompletionStreamOptionsBase<ChatCompletionStreamOptions> {
    // specialized methods if needed
};

struct ChatCompletionStreamResponseDelta : public ChatCompletionStreamResponseDeltaBase<ChatCompletionStreamResponseDelta> {
    // specialized methods if needed
};

struct ChatCompletionTokenLogprob : public ChatCompletionTokenLogprobBase<ChatCompletionTokenLogprob> {
    // specialized methods if needed
};

struct ChatCompletionTool : public ChatCompletionToolBase<ChatCompletionTool> {
    // specialized methods if needed
};

struct ChatCompletionToolChoiceOption : public ChatCompletionToolChoiceOptionBase<ChatCompletionToolChoiceOption> {
    // specialized methods if needed
};

struct ChunkingStrategyRequestParam : public ChunkingStrategyRequestParamBase<ChunkingStrategyRequestParam> {
    // specialized methods if needed
};

struct CompleteUploadRequest : public CompleteUploadRequestBase<CompleteUploadRequest> {
    // specialized methods if needed
};

struct CompletionUsage : public CompletionUsageBase<CompletionUsage> {
    // specialized methods if needed
};

struct CostsResult : public CostsResultBase<CostsResult> {
    // specialized methods if needed
};

struct CreateAssistantRequest : public CreateAssistantRequestBase<CreateAssistantRequest> {
    // specialized methods if needed
};

struct CreateChatCompletionFunctionResponse : public CreateChatCompletionFunctionResponseBase<CreateChatCompletionFunctionResponse> {
    // specialized methods if needed
};

struct CreateChatCompletionImageResponse : public CreateChatCompletionImageResponseBase<CreateChatCompletionImageResponse> {
    // specialized methods if needed
};

struct ParallelToolCalls : public ParallelToolCallsBase<ParallelToolCalls> {
    // specialized methods if needed
};

struct CreateChatCompletionRequest : public CreateChatCompletionRequestBase<CreateChatCompletionRequest> {
    // specialized methods if needed
};

struct CreateChatCompletionResponse : public CreateChatCompletionResponseBase<CreateChatCompletionResponse> {
    // specialized methods if needed
};

struct CreateChatCompletionStreamResponse : public CreateChatCompletionStreamResponseBase<CreateChatCompletionStreamResponse> {
    // specialized methods if needed
};

struct CreateCompletionRequest : public CreateCompletionRequestBase<CreateCompletionRequest> {
    // specialized methods if needed
};

struct CreateCompletionResponse : public CreateCompletionResponseBase<CreateCompletionResponse> {
    // specialized methods if needed
};

struct CreateEmbeddingRequest : public CreateEmbeddingRequestBase<CreateEmbeddingRequest> {
    // specialized methods if needed
};

struct CreateEmbeddingResponse : public CreateEmbeddingResponseBase<CreateEmbeddingResponse> {
    // specialized methods if needed
};

struct CreateFileRequest : public CreateFileRequestBase<CreateFileRequest> {
    // specialized methods if needed
};

struct FineTuneSupervisedMethod : public FineTuneSupervisedMethodBase<FineTuneSupervisedMethod> {
    // specialized methods if needed
};

struct FineTuneDPOMethod : public FineTuneDPOMethodBase<FineTuneDPOMethod> {
    // specialized methods if needed
};

struct FineTuneMethod : public FineTuneMethodBase<FineTuneMethod> {
    // specialized methods if needed
};

struct CreateFineTuningJobRequest : public CreateFineTuningJobRequestBase<CreateFineTuningJobRequest> {
    // specialized methods if needed
};

struct CreateImageEditRequest : public CreateImageEditRequestBase<CreateImageEditRequest> {
    // specialized methods if needed
};

struct CreateImageRequest : public CreateImageRequestBase<CreateImageRequest> {
    // specialized methods if needed
};

struct CreateImageVariationRequest : public CreateImageVariationRequestBase<CreateImageVariationRequest> {
    // specialized methods if needed
};

struct CreateMessageRequest : public CreateMessageRequestBase<CreateMessageRequest> {
    // specialized methods if needed
};

struct CreateModerationRequest : public CreateModerationRequestBase<CreateModerationRequest> {
    // specialized methods if needed
};

struct CreateModerationResponse : public CreateModerationResponseBase<CreateModerationResponse> {
    // specialized methods if needed
};

struct TruncationObject : public TruncationObjectBase<TruncationObject> {
    // specialized methods if needed
};

struct CreateRunRequest : public CreateRunRequestBase<CreateRunRequest> {
    // specialized methods if needed
};

struct CreateSpeechRequest : public CreateSpeechRequestBase<CreateSpeechRequest> {
    // specialized methods if needed
};

struct CreateThreadRequest : public CreateThreadRequestBase<CreateThreadRequest> {
    // specialized methods if needed
};

struct CreateThreadAndRunRequest : public CreateThreadAndRunRequestBase<CreateThreadAndRunRequest> {
    // specialized methods if needed
};

struct CreateTranscriptionRequest : public CreateTranscriptionRequestBase<CreateTranscriptionRequest> {
    // specialized methods if needed
};

struct CreateTranscriptionResponseJson : public CreateTranscriptionResponseJsonBase<CreateTranscriptionResponseJson> {
    // specialized methods if needed
};

struct CreateTranscriptionResponseVerboseJson : public CreateTranscriptionResponseVerboseJsonBase<CreateTranscriptionResponseVerboseJson> {
    // specialized methods if needed
};

struct CreateTranslationRequest : public CreateTranslationRequestBase<CreateTranslationRequest> {
    // specialized methods if needed
};

struct CreateTranslationResponseJson : public CreateTranslationResponseJsonBase<CreateTranslationResponseJson> {
    // specialized methods if needed
};

struct CreateTranslationResponseVerboseJson : public CreateTranslationResponseVerboseJsonBase<CreateTranslationResponseVerboseJson> {
    // specialized methods if needed
};

struct CreateUploadRequest : public CreateUploadRequestBase<CreateUploadRequest> {
    // specialized methods if needed
};

struct CreateVectorStoreFileBatchRequest : public CreateVectorStoreFileBatchRequestBase<CreateVectorStoreFileBatchRequest> {
    // specialized methods if needed
};

struct CreateVectorStoreFileRequest : public CreateVectorStoreFileRequestBase<CreateVectorStoreFileRequest> {
    // specialized methods if needed
};

struct VectorStoreExpirationAfter : public VectorStoreExpirationAfterBase<VectorStoreExpirationAfter> {
    // specialized methods if needed
};

struct CreateVectorStoreRequest : public CreateVectorStoreRequestBase<CreateVectorStoreRequest> {
    // specialized methods if needed
};

struct DefaultProjectErrorResponse : public DefaultProjectErrorResponseBase<DefaultProjectErrorResponse> {
    // specialized methods if needed
};

struct DeleteAssistantResponse : public DeleteAssistantResponseBase<DeleteAssistantResponse> {
    // specialized methods if needed
};

struct DeleteFileResponse : public DeleteFileResponseBase<DeleteFileResponse> {
    // specialized methods if needed
};

struct DeleteMessageResponse : public DeleteMessageResponseBase<DeleteMessageResponse> {
    // specialized methods if needed
};

struct DeleteModelResponse : public DeleteModelResponseBase<DeleteModelResponse> {
    // specialized methods if needed
};

struct DeleteThreadResponse : public DeleteThreadResponseBase<DeleteThreadResponse> {
    // specialized methods if needed
};

struct DeleteVectorStoreFileResponse : public DeleteVectorStoreFileResponseBase<DeleteVectorStoreFileResponse> {
    // specialized methods if needed
};

struct DeleteVectorStoreResponse : public DeleteVectorStoreResponseBase<DeleteVectorStoreResponse> {
    // specialized methods if needed
};

struct DoneEvent : public DoneEventBase<DoneEvent> {
    // specialized methods if needed
};

struct Embedding : public EmbeddingBase<Embedding> {
    // specialized methods if needed
};

struct Error : public ErrorBase<Error> {
    // specialized methods if needed
};

struct ErrorEvent : public ErrorEventBase<ErrorEvent> {
    // specialized methods if needed
};

struct ErrorResponse : public ErrorResponseBase<ErrorResponse> {
    // specialized methods if needed
};

struct FileSearchRankingOptions : public FileSearchRankingOptionsBase<FileSearchRankingOptions> {
    // specialized methods if needed
};

struct FineTuneChatCompletionRequestAssistantMessage : public FineTuneChatCompletionRequestAssistantMessageBase<FineTuneChatCompletionRequestAssistantMessage> {
    // specialized methods if needed
};

struct FineTuneChatRequestInput : public FineTuneChatRequestInputBase<FineTuneChatRequestInput> {
    // specialized methods if needed
};

struct FineTuneCompletionRequestInput : public FineTuneCompletionRequestInputBase<FineTuneCompletionRequestInput> {
    // specialized methods if needed
};

struct FineTunePreferenceRequestInput : public FineTunePreferenceRequestInputBase<FineTunePreferenceRequestInput> {
    // specialized methods if needed
};

struct FineTuningIntegration : public FineTuningIntegrationBase<FineTuningIntegration> {
    // specialized methods if needed
};

struct FineTuningJob : public FineTuningJobBase<FineTuningJob> {
    // specialized methods if needed
};

struct FineTuningJobCheckpoint : public FineTuningJobCheckpointBase<FineTuningJobCheckpoint> {
    // specialized methods if needed
};

struct FineTuningJobEvent : public FineTuningJobEventBase<FineTuningJobEvent> {
    // specialized methods if needed
};

struct Image : public ImageBase<Image> {
    // specialized methods if needed
};

struct ImagesResponse : public ImagesResponseBase<ImagesResponse> {
    // specialized methods if needed
};

struct Invite : public InviteBase<Invite> {
    // specialized methods if needed
};

struct InviteDeleteResponse : public InviteDeleteResponseBase<InviteDeleteResponse> {
    // specialized methods if needed
};

struct InviteListResponse : public InviteListResponseBase<InviteListResponse> {
    // specialized methods if needed
};

struct InviteRequest : public InviteRequestBase<InviteRequest> {
    // specialized methods if needed
};

struct ListAssistantsResponse : public ListAssistantsResponseBase<ListAssistantsResponse> {
    // specialized methods if needed
};

struct ListAuditLogsResponse : public ListAuditLogsResponseBase<ListAuditLogsResponse> {
    // specialized methods if needed
};

struct ListBatchesResponse : public ListBatchesResponseBase<ListBatchesResponse> {
    // specialized methods if needed
};

struct ListFilesResponse : public ListFilesResponseBase<ListFilesResponse> {
    // specialized methods if needed
};

struct ListFineTuningJobCheckpointsResponse : public ListFineTuningJobCheckpointsResponseBase<ListFineTuningJobCheckpointsResponse> {
    // specialized methods if needed
};

struct ListFineTuningJobEventsResponse : public ListFineTuningJobEventsResponseBase<ListFineTuningJobEventsResponse> {
    // specialized methods if needed
};

struct ListMessagesResponse : public ListMessagesResponseBase<ListMessagesResponse> {
    // specialized methods if needed
};

struct ListModelsResponse : public ListModelsResponseBase<ListModelsResponse> {
    // specialized methods if needed
};

struct ListPaginatedFineTuningJobsResponse : public ListPaginatedFineTuningJobsResponseBase<ListPaginatedFineTuningJobsResponse> {
    // specialized methods if needed
};

struct ListRunStepsResponse : public ListRunStepsResponseBase<ListRunStepsResponse> {
    // specialized methods if needed
};

struct ListRunsResponse : public ListRunsResponseBase<ListRunsResponse> {
    // specialized methods if needed
};

struct ListThreadsResponse : public ListThreadsResponseBase<ListThreadsResponse> {
    // specialized methods if needed
};

struct ListVectorStoreFilesResponse : public ListVectorStoreFilesResponseBase<ListVectorStoreFilesResponse> {
    // specialized methods if needed
};

struct ListVectorStoresResponse : public ListVectorStoresResponseBase<ListVectorStoresResponse> {
    // specialized methods if needed
};

struct MessageContentImageFileObject : public MessageContentImageFileObjectBase<MessageContentImageFileObject> {
    // specialized methods if needed
};

struct MessageContentImageUrlObject : public MessageContentImageUrlObjectBase<MessageContentImageUrlObject> {
    // specialized methods if needed
};

struct MessageContentRefusalObject : public MessageContentRefusalObjectBase<MessageContentRefusalObject> {
    // specialized methods if needed
};

struct MessageContentTextAnnotationsFileCitationObject : public MessageContentTextAnnotationsFileCitationObjectBase<MessageContentTextAnnotationsFileCitationObject> {
    // specialized methods if needed
};

struct MessageContentTextAnnotationsFilePathObject : public MessageContentTextAnnotationsFilePathObjectBase<MessageContentTextAnnotationsFilePathObject> {
    // specialized methods if needed
};

struct MessageContentTextObject : public MessageContentTextObjectBase<MessageContentTextObject> {
    // specialized methods if needed
};

struct MessageDeltaContentImageFileObject : public MessageDeltaContentImageFileObjectBase<MessageDeltaContentImageFileObject> {
    // specialized methods if needed
};

struct MessageDeltaContentImageUrlObject : public MessageDeltaContentImageUrlObjectBase<MessageDeltaContentImageUrlObject> {
    // specialized methods if needed
};

struct MessageDeltaContentRefusalObject : public MessageDeltaContentRefusalObjectBase<MessageDeltaContentRefusalObject> {
    // specialized methods if needed
};

struct MessageDeltaContentTextAnnotationsFileCitationObject : public MessageDeltaContentTextAnnotationsFileCitationObjectBase<MessageDeltaContentTextAnnotationsFileCitationObject> {
    // specialized methods if needed
};

struct MessageDeltaContentTextAnnotationsFilePathObject : public MessageDeltaContentTextAnnotationsFilePathObjectBase<MessageDeltaContentTextAnnotationsFilePathObject> {
    // specialized methods if needed
};

struct MessageDeltaContentTextObject : public MessageDeltaContentTextObjectBase<MessageDeltaContentTextObject> {
    // specialized methods if needed
};

struct MessageDeltaObject : public MessageDeltaObjectBase<MessageDeltaObject> {
    // specialized methods if needed
};

struct MessageObject : public MessageObjectBase<MessageObject> {
    // specialized methods if needed
};

struct MessageRequestContentTextObject : public MessageRequestContentTextObjectBase<MessageRequestContentTextObject> {
    // specialized methods if needed
};

struct MessageStreamEvent : public MessageStreamEventBase<MessageStreamEvent> {
    // specialized methods if needed
};

struct Model : public ModelBase<Model> {
    // specialized methods if needed
};

struct ModifyAssistantRequest : public ModifyAssistantRequestBase<ModifyAssistantRequest> {
    // specialized methods if needed
};

struct ModifyMessageRequest : public ModifyMessageRequestBase<ModifyMessageRequest> {
    // specialized methods if needed
};

struct ModifyRunRequest : public ModifyRunRequestBase<ModifyRunRequest> {
    // specialized methods if needed
};

struct ModifyThreadRequest : public ModifyThreadRequestBase<ModifyThreadRequest> {
    // specialized methods if needed
};

struct OpenAIFile : public OpenAIFileBase<OpenAIFile> {
    // specialized methods if needed
};

struct OtherChunkingStrategyResponseParam : public OtherChunkingStrategyResponseParamBase<OtherChunkingStrategyResponseParam> {
    // specialized methods if needed
};

struct PredictionContent : public PredictionContentBase<PredictionContent> {
    // specialized methods if needed
};

struct Project : public ProjectBase<Project> {
    // specialized methods if needed
};

struct ProjectApiKey : public ProjectApiKeyBase<ProjectApiKey> {
    // specialized methods if needed
};

struct ProjectApiKeyDeleteResponse : public ProjectApiKeyDeleteResponseBase<ProjectApiKeyDeleteResponse> {
    // specialized methods if needed
};

struct ProjectApiKeyListResponse : public ProjectApiKeyListResponseBase<ProjectApiKeyListResponse> {
    // specialized methods if needed
};

struct ProjectCreateRequest : public ProjectCreateRequestBase<ProjectCreateRequest> {
    // specialized methods if needed
};

struct ProjectListResponse : public ProjectListResponseBase<ProjectListResponse> {
    // specialized methods if needed
};

struct ProjectRateLimit : public ProjectRateLimitBase<ProjectRateLimit> {
    // specialized methods if needed
};

struct ProjectRateLimitListResponse : public ProjectRateLimitListResponseBase<ProjectRateLimitListResponse> {
    // specialized methods if needed
};

struct ProjectRateLimitUpdateRequest : public ProjectRateLimitUpdateRequestBase<ProjectRateLimitUpdateRequest> {
    // specialized methods if needed
};

struct ProjectServiceAccount : public ProjectServiceAccountBase<ProjectServiceAccount> {
    // specialized methods if needed
};

struct ProjectServiceAccountApiKey : public ProjectServiceAccountApiKeyBase<ProjectServiceAccountApiKey> {
    // specialized methods if needed
};

struct ProjectServiceAccountCreateRequest : public ProjectServiceAccountCreateRequestBase<ProjectServiceAccountCreateRequest> {
    // specialized methods if needed
};

struct ProjectServiceAccountCreateResponse : public ProjectServiceAccountCreateResponseBase<ProjectServiceAccountCreateResponse> {
    // specialized methods if needed
};

struct ProjectServiceAccountDeleteResponse : public ProjectServiceAccountDeleteResponseBase<ProjectServiceAccountDeleteResponse> {
    // specialized methods if needed
};

struct ProjectServiceAccountListResponse : public ProjectServiceAccountListResponseBase<ProjectServiceAccountListResponse> {
    // specialized methods if needed
};

struct ProjectUpdateRequest : public ProjectUpdateRequestBase<ProjectUpdateRequest> {
    // specialized methods if needed
};

struct ProjectUser : public ProjectUserBase<ProjectUser> {
    // specialized methods if needed
};

struct ProjectUserCreateRequest : public ProjectUserCreateRequestBase<ProjectUserCreateRequest> {
    // specialized methods if needed
};

struct ProjectUserDeleteResponse : public ProjectUserDeleteResponseBase<ProjectUserDeleteResponse> {
    // specialized methods if needed
};

struct ProjectUserListResponse : public ProjectUserListResponseBase<ProjectUserListResponse> {
    // specialized methods if needed
};

struct ProjectUserUpdateRequest : public ProjectUserUpdateRequestBase<ProjectUserUpdateRequest> {
    // specialized methods if needed
};

struct RealtimeConversationItem : public RealtimeConversationItemBase<RealtimeConversationItem> {
    // specialized methods if needed
};

struct RealtimeClientEventConversationItemCreate : public RealtimeClientEventConversationItemCreateBase<RealtimeClientEventConversationItemCreate> {
    // specialized methods if needed
};

struct RealtimeClientEventConversationItemDelete : public RealtimeClientEventConversationItemDeleteBase<RealtimeClientEventConversationItemDelete> {
    // specialized methods if needed
};

struct RealtimeClientEventConversationItemTruncate : public RealtimeClientEventConversationItemTruncateBase<RealtimeClientEventConversationItemTruncate> {
    // specialized methods if needed
};

struct RealtimeClientEventInputAudioBufferAppend : public RealtimeClientEventInputAudioBufferAppendBase<RealtimeClientEventInputAudioBufferAppend> {
    // specialized methods if needed
};

struct RealtimeClientEventInputAudioBufferClear : public RealtimeClientEventInputAudioBufferClearBase<RealtimeClientEventInputAudioBufferClear> {
    // specialized methods if needed
};

struct RealtimeClientEventInputAudioBufferCommit : public RealtimeClientEventInputAudioBufferCommitBase<RealtimeClientEventInputAudioBufferCommit> {
    // specialized methods if needed
};

struct RealtimeClientEventResponseCancel : public RealtimeClientEventResponseCancelBase<RealtimeClientEventResponseCancel> {
    // specialized methods if needed
};

struct RealtimeResponseCreateParams : public RealtimeResponseCreateParamsBase<RealtimeResponseCreateParams> {
    // specialized methods if needed
};

struct RealtimeClientEventResponseCreate : public RealtimeClientEventResponseCreateBase<RealtimeClientEventResponseCreate> {
    // specialized methods if needed
};

struct RealtimeSessionCreateRequest : public RealtimeSessionCreateRequestBase<RealtimeSessionCreateRequest> {
    // specialized methods if needed
};

struct RealtimeClientEventSessionUpdate : public RealtimeClientEventSessionUpdateBase<RealtimeClientEventSessionUpdate> {
    // specialized methods if needed
};

struct RealtimeResponse : public RealtimeResponseBase<RealtimeResponse> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationCreated : public RealtimeServerEventConversationCreatedBase<RealtimeServerEventConversationCreated> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationItemCreated : public RealtimeServerEventConversationItemCreatedBase<RealtimeServerEventConversationItemCreated> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationItemDeleted : public RealtimeServerEventConversationItemDeletedBase<RealtimeServerEventConversationItemDeleted> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationItemInputAudioTranscriptionCompleted : public RealtimeServerEventConversationItemInputAudioTranscriptionCompletedBase<RealtimeServerEventConversationItemInputAudioTranscriptionCompleted> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationItemInputAudioTranscriptionFailed : public RealtimeServerEventConversationItemInputAudioTranscriptionFailedBase<RealtimeServerEventConversationItemInputAudioTranscriptionFailed> {
    // specialized methods if needed
};

struct RealtimeServerEventConversationItemTruncated : public RealtimeServerEventConversationItemTruncatedBase<RealtimeServerEventConversationItemTruncated> {
    // specialized methods if needed
};

struct RealtimeServerEventError : public RealtimeServerEventErrorBase<RealtimeServerEventError> {
    // specialized methods if needed
};

struct RealtimeServerEventInputAudioBufferCleared : public RealtimeServerEventInputAudioBufferClearedBase<RealtimeServerEventInputAudioBufferCleared> {
    // specialized methods if needed
};

struct RealtimeServerEventInputAudioBufferCommitted : public RealtimeServerEventInputAudioBufferCommittedBase<RealtimeServerEventInputAudioBufferCommitted> {
    // specialized methods if needed
};

struct RealtimeServerEventInputAudioBufferSpeechStarted : public RealtimeServerEventInputAudioBufferSpeechStartedBase<RealtimeServerEventInputAudioBufferSpeechStarted> {
    // specialized methods if needed
};

struct RealtimeServerEventInputAudioBufferSpeechStopped : public RealtimeServerEventInputAudioBufferSpeechStoppedBase<RealtimeServerEventInputAudioBufferSpeechStopped> {
    // specialized methods if needed
};

struct RealtimeServerEventRateLimitsUpdated : public RealtimeServerEventRateLimitsUpdatedBase<RealtimeServerEventRateLimitsUpdated> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseAudioDelta : public RealtimeServerEventResponseAudioDeltaBase<RealtimeServerEventResponseAudioDelta> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseAudioDone : public RealtimeServerEventResponseAudioDoneBase<RealtimeServerEventResponseAudioDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseAudioTranscriptDelta : public RealtimeServerEventResponseAudioTranscriptDeltaBase<RealtimeServerEventResponseAudioTranscriptDelta> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseAudioTranscriptDone : public RealtimeServerEventResponseAudioTranscriptDoneBase<RealtimeServerEventResponseAudioTranscriptDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseContentPartAdded : public RealtimeServerEventResponseContentPartAddedBase<RealtimeServerEventResponseContentPartAdded> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseContentPartDone : public RealtimeServerEventResponseContentPartDoneBase<RealtimeServerEventResponseContentPartDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseCreated : public RealtimeServerEventResponseCreatedBase<RealtimeServerEventResponseCreated> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseDone : public RealtimeServerEventResponseDoneBase<RealtimeServerEventResponseDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseFunctionCallArgumentsDelta : public RealtimeServerEventResponseFunctionCallArgumentsDeltaBase<RealtimeServerEventResponseFunctionCallArgumentsDelta> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseFunctionCallArgumentsDone : public RealtimeServerEventResponseFunctionCallArgumentsDoneBase<RealtimeServerEventResponseFunctionCallArgumentsDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseOutputItemAdded : public RealtimeServerEventResponseOutputItemAddedBase<RealtimeServerEventResponseOutputItemAdded> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseOutputItemDone : public RealtimeServerEventResponseOutputItemDoneBase<RealtimeServerEventResponseOutputItemDone> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseTextDelta : public RealtimeServerEventResponseTextDeltaBase<RealtimeServerEventResponseTextDelta> {
    // specialized methods if needed
};

struct RealtimeServerEventResponseTextDone : public RealtimeServerEventResponseTextDoneBase<RealtimeServerEventResponseTextDone> {
    // specialized methods if needed
};

struct RealtimeSession : public RealtimeSessionBase<RealtimeSession> {
    // specialized methods if needed
};

struct RealtimeServerEventSessionCreated : public RealtimeServerEventSessionCreatedBase<RealtimeServerEventSessionCreated> {
    // specialized methods if needed
};

struct RealtimeServerEventSessionUpdated : public RealtimeServerEventSessionUpdatedBase<RealtimeServerEventSessionUpdated> {
    // specialized methods if needed
};

struct RealtimeSessionCreateResponse : public RealtimeSessionCreateResponseBase<RealtimeSessionCreateResponse> {
    // specialized methods if needed
};

struct ResponseFormatJsonObject : public ResponseFormatJsonObjectBase<ResponseFormatJsonObject> {
    // specialized methods if needed
};

struct ResponseFormatJsonSchema : public ResponseFormatJsonSchemaBase<ResponseFormatJsonSchema> {
    // specialized methods if needed
};

struct ResponseFormatJsonSchemaSchema : public ResponseFormatJsonSchemaSchemaBase<ResponseFormatJsonSchemaSchema> {
    // specialized methods if needed
};

struct ResponseFormatText : public ResponseFormatTextBase<ResponseFormatText> {
    // specialized methods if needed
};

struct RunCompletionUsage : public RunCompletionUsageBase<RunCompletionUsage> {
    // specialized methods if needed
};

struct RunObject : public RunObjectBase<RunObject> {
    // specialized methods if needed
};

struct RunStepCompletionUsage : public RunStepCompletionUsageBase<RunStepCompletionUsage> {
    // specialized methods if needed
};

struct RunStepDeltaObject : public RunStepDeltaObjectBase<RunStepDeltaObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsMessageCreationObject : public RunStepDeltaStepDetailsMessageCreationObjectBase<RunStepDeltaStepDetailsMessageCreationObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsCodeObject : public RunStepDeltaStepDetailsToolCallsCodeObjectBase<RunStepDeltaStepDetailsToolCallsCodeObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsCodeOutputImageObject : public RunStepDeltaStepDetailsToolCallsCodeOutputImageObjectBase<RunStepDeltaStepDetailsToolCallsCodeOutputImageObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject : public RunStepDeltaStepDetailsToolCallsCodeOutputLogsObjectBase<RunStepDeltaStepDetailsToolCallsCodeOutputLogsObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsFileSearchObject : public RunStepDeltaStepDetailsToolCallsFileSearchObjectBase<RunStepDeltaStepDetailsToolCallsFileSearchObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsFunctionObject : public RunStepDeltaStepDetailsToolCallsFunctionObjectBase<RunStepDeltaStepDetailsToolCallsFunctionObject> {
    // specialized methods if needed
};

struct RunStepDeltaStepDetailsToolCallsObject : public RunStepDeltaStepDetailsToolCallsObjectBase<RunStepDeltaStepDetailsToolCallsObject> {
    // specialized methods if needed
};

struct RunStepDetailsMessageCreationObject : public RunStepDetailsMessageCreationObjectBase<RunStepDetailsMessageCreationObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsCodeObject : public RunStepDetailsToolCallsCodeObjectBase<RunStepDetailsToolCallsCodeObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsCodeOutputImageObject : public RunStepDetailsToolCallsCodeOutputImageObjectBase<RunStepDetailsToolCallsCodeOutputImageObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsCodeOutputLogsObject : public RunStepDetailsToolCallsCodeOutputLogsObjectBase<RunStepDetailsToolCallsCodeOutputLogsObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsFileSearchObject : public RunStepDetailsToolCallsFileSearchObjectBase<RunStepDetailsToolCallsFileSearchObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsFileSearchRankingOptionsObject : public RunStepDetailsToolCallsFileSearchRankingOptionsObjectBase<RunStepDetailsToolCallsFileSearchRankingOptionsObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsFileSearchResultObject : public RunStepDetailsToolCallsFileSearchResultObjectBase<RunStepDetailsToolCallsFileSearchResultObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsFunctionObject : public RunStepDetailsToolCallsFunctionObjectBase<RunStepDetailsToolCallsFunctionObject> {
    // specialized methods if needed
};

struct RunStepDetailsToolCallsObject : public RunStepDetailsToolCallsObjectBase<RunStepDetailsToolCallsObject> {
    // specialized methods if needed
};

struct RunStepObject : public RunStepObjectBase<RunStepObject> {
    // specialized methods if needed
};

struct RunStepStreamEvent : public RunStepStreamEventBase<RunStepStreamEvent> {
    // specialized methods if needed
};

struct RunStreamEvent : public RunStreamEventBase<RunStreamEvent> {
    // specialized methods if needed
};

struct RunToolCallObject : public RunToolCallObjectBase<RunToolCallObject> {
    // specialized methods if needed
};

struct StaticChunkingStrategy : public StaticChunkingStrategyBase<StaticChunkingStrategy> {
    // specialized methods if needed
};

struct StaticChunkingStrategyRequestParam : public StaticChunkingStrategyRequestParamBase<StaticChunkingStrategyRequestParam> {
    // specialized methods if needed
};

struct StaticChunkingStrategyResponseParam : public StaticChunkingStrategyResponseParamBase<StaticChunkingStrategyResponseParam> {
    // specialized methods if needed
};

struct SubmitToolOutputsRunRequest : public SubmitToolOutputsRunRequestBase<SubmitToolOutputsRunRequest> {
    // specialized methods if needed
};

struct ThreadObject : public ThreadObjectBase<ThreadObject> {
    // specialized methods if needed
};

struct ThreadStreamEvent : public ThreadStreamEventBase<ThreadStreamEvent> {
    // specialized methods if needed
};

struct TranscriptionSegment : public TranscriptionSegmentBase<TranscriptionSegment> {
    // specialized methods if needed
};

struct TranscriptionWord : public TranscriptionWordBase<TranscriptionWord> {
    // specialized methods if needed
};

struct UpdateVectorStoreRequest : public UpdateVectorStoreRequestBase<UpdateVectorStoreRequest> {
    // specialized methods if needed
};

struct Upload : public UploadBase<Upload> {
    // specialized methods if needed
};

struct UploadPart : public UploadPartBase<UploadPart> {
    // specialized methods if needed
};

struct UsageAudioSpeechesResult : public UsageAudioSpeechesResultBase<UsageAudioSpeechesResult> {
    // specialized methods if needed
};

struct UsageAudioTranscriptionsResult : public UsageAudioTranscriptionsResultBase<UsageAudioTranscriptionsResult> {
    // specialized methods if needed
};

struct UsageCodeInterpreterSessionsResult : public UsageCodeInterpreterSessionsResultBase<UsageCodeInterpreterSessionsResult> {
    // specialized methods if needed
};

struct UsageCompletionsResult : public UsageCompletionsResultBase<UsageCompletionsResult> {
    // specialized methods if needed
};

struct UsageEmbeddingsResult : public UsageEmbeddingsResultBase<UsageEmbeddingsResult> {
    // specialized methods if needed
};

struct UsageImagesResult : public UsageImagesResultBase<UsageImagesResult> {
    // specialized methods if needed
};

struct UsageModerationsResult : public UsageModerationsResultBase<UsageModerationsResult> {
    // specialized methods if needed
};

struct UsageResponse : public UsageResponseBase<UsageResponse> {
    // specialized methods if needed
};

struct UsageTimeBucket : public UsageTimeBucketBase<UsageTimeBucket> {
    // specialized methods if needed
};

struct UsageVectorStoresResult : public UsageVectorStoresResultBase<UsageVectorStoresResult> {
    // specialized methods if needed
};

struct User : public UserBase<User> {
    // specialized methods if needed
};

struct UserDeleteResponse : public UserDeleteResponseBase<UserDeleteResponse> {
    // specialized methods if needed
};

struct UserListResponse : public UserListResponseBase<UserListResponse> {
    // specialized methods if needed
};

struct UserRoleUpdateRequest : public UserRoleUpdateRequestBase<UserRoleUpdateRequest> {
    // specialized methods if needed
};

struct VectorStoreFileBatchObject : public VectorStoreFileBatchObjectBase<VectorStoreFileBatchObject> {
    // specialized methods if needed
};

struct VectorStoreFileObject : public VectorStoreFileObjectBase<VectorStoreFileObject> {
    // specialized methods if needed
};

struct VectorStoreObject : public VectorStoreObjectBase<VectorStoreObject> {
    // specialized methods if needed
};
