// lib/src/api_service.dart
import 'package:dio/dio.dart';
import 'package:document_chat/src/models/page_search_result.dart';
import 'package:file_picker/file_picker.dart';
import 'package:document_chat/src/models/document.dart';
import 'package:document_chat/src/models/processing_progress.dart';

class ApiException implements Exception {
  final String message;
  final String? errorCode;

  ApiException(this.message, {this.errorCode});

  String get userMessage {
    switch (errorCode) {
      case 'FILE_TOO_LARGE':
        return 'File is too large. Maximum size is 50MB.';
      case 'INVALID_FORMAT':
        return 'Invalid file format. Please upload PDF, JPG, or PNG files.';
      case 'DUPLICATE_FILE':
        return 'This document has already been uploaded.';
      case 'NO_TEXT_FOUND':
        return 'Could not extract text from this document.';
      case 'OCR_TIMEOUT':
        return 'Processing timed out. Try a smaller or clearer document.';
      case 'PROCESSING_FAILED':
        return 'Processing failed. Please try again.';
      default:
        return message;
    }
  }

  @override
  String toString() => userMessage;
}

class ApiService {
  final Dio _dio;

  ApiService()
      : _dio = Dio(BaseOptions(
          baseUrl: 'http://100.127.26.110:8000',
          connectTimeout: const Duration(seconds: 10),
          receiveTimeout: const Duration(seconds: 60),
        ));

  Never _handleDioError(DioException e) {
    final errorCode = e.response?.statusCode ?? 0;
    String? apiErrorCode;

    if (e.response?.data is Map && e.response?.data['error_code'] != null) {
      apiErrorCode = e.response?.data['error_code'];
    }

    if (e.response?.data is Map && e.response?.data['detail'] != null) {
      final detail = e.response?.data['detail'];
      throw ApiException('[$errorCode] $detail', errorCode: apiErrorCode);
    }

    switch (e.type) {
      case DioExceptionType.connectionTimeout:
      case DioExceptionType.sendTimeout:
      case DioExceptionType.receiveTimeout:
        throw ApiException(
          '[$errorCode] Connection timed out. Please try again.',
          errorCode: 'TIMEOUT',
        );
      case DioExceptionType.connectionError:
        throw ApiException(
          '[$errorCode] Could not connect to server. Check network.',
          errorCode: 'CONNECTION_ERROR',
        );
      default:
        throw ApiException(
          '[$errorCode] Unexpected error occurred.',
          errorCode: 'UNKNOWN',
        );
    }
  }

  // Documents
  Future<List<Document>> listDocuments() async {
    try {
      final response = await _dio.get('/documents');
      final List<dynamic> docList = response.data['documents'];
      return docList.map((json) => Document.fromJson(json)).toList();
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  Future<void> deleteDocument(String docId) async {
    try {
      await _dio.delete('/documents/$docId');
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  Future<void> clearAllDocuments() async {
    try {
      await _dio.delete('/documents');
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  // Upload
  Future<String> uploadDocument(PlatformFile file) async {
    MultipartFile multipartFile;
    if (file.bytes != null) {
      multipartFile = MultipartFile.fromBytes(file.bytes!, filename: file.name);
    } else if (file.path != null) {
      multipartFile =
          await MultipartFile.fromFile(file.path!, filename: file.name);
    } else {
      throw ApiException('Cannot read the selected file.');
    }

    final formData = FormData.fromMap({'file': multipartFile});

    try {
      final response = await _dio.post('/upload-document', data: formData);

      if (response.data['status'] == 'error') {
        throw ApiException(
          response.data['error'] ?? 'Upload failed',
          errorCode: response.data['error_code'],
        );
      }

      return response.data['document_id'] as String;
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  Future<ProcessingProgress> getProcessingStatus(String documentId) async {
    try {
      final response = await _dio.get('/processing-status/$documentId');
      return ProcessingProgress.fromJson(response.data);
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  // Search - Page-based (PRIMARY)
  Future<List<PageSearchResult>> searchPages(String query) async {
    try {
      final response =
          await _dio.post('/search-pages', data: {'question': query});

      final List<dynamic> results = response.data['results'];
      return results.map((json) => PageSearchResult.fromJson(json)).toList();
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  // Default search (alias)
  Future<List<PageSearchResult>> search(String query) async {
    return searchPages(query);
  }

  // Chat History
  Future<List<Map<String, dynamic>>> getChatHistory() async {
    try {
      final response = await _dio.get('/search-history');
      return List<Map<String, dynamic>>.from(response.data);
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }

  Future<void> clearChatHistory() async {
    try {
      await _dio.delete('/search-history');
    } on DioException catch (e) {
      _handleDioError(e);
    }
  }
}
