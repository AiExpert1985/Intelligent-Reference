// lib/src/models/processing_progress.dart
enum ProcessingStatus {
  pending,
  validating,
  extractingText,
  generatingEmbeddings,
  storing,
  completed,
  failed;

  static ProcessingStatus fromString(String status) {
    switch (status) {
      case 'pending':
        return ProcessingStatus.pending;
      case 'validating':
        return ProcessingStatus.validating;
      case 'extracting_text':
        return ProcessingStatus.extractingText;
      case 'generating_embeddings':
        return ProcessingStatus.generatingEmbeddings;
      case 'storing':
        return ProcessingStatus.storing;
      case 'completed':
        return ProcessingStatus.completed;
      case 'failed':
        return ProcessingStatus.failed;
      default:
        return ProcessingStatus.failed; // Safer default
    }
  }
}

class ProcessingProgress {
  final String documentId;
  final String filename;
  final ProcessingStatus status;
  final int progressPercent;
  final String currentStep;
  final String? error;
  final String? errorCode;

  ProcessingProgress({
    required this.documentId,
    required this.filename,
    required this.status,
    required this.progressPercent,
    required this.currentStep,
    this.error,
    this.errorCode,
  });

  factory ProcessingProgress.fromJson(Map<String, dynamic> json) {
    return ProcessingProgress(
      documentId: json['document_id'] as String,
      filename: json['filename'] as String,
      status: ProcessingStatus.fromString(json['status'] as String),
      progressPercent: json['progress_percent'] as int,
      currentStep: json['current_step'] as String,
      error: json['error'] as String?,
      errorCode: json['error_code'] as String?,
    );
  }

  bool get isComplete => status == ProcessingStatus.completed;
  bool get hasFailed => status == ProcessingStatus.failed;
}
