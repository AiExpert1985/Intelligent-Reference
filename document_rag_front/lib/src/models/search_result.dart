// lib/src/models/search_result.dart
class SearchResult {
  final String documentName;
  final int pageNumber;
  final String contentSnippet;
  final String documentId;
  final String downloadUrl; // ADD THIS LINE

  SearchResult({
    required this.documentName,
    required this.pageNumber,
    required this.contentSnippet,
    required this.documentId,
    required this.downloadUrl, // ADD THIS LINE
  });

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      documentName: json['document_name'] as String,
      pageNumber: json['page_number'] as int,
      contentSnippet: json['content_snippet'] as String,
      documentId: json['document_id'] as String,
      downloadUrl: json['download_url'] as String, // ADD THIS LINE
    );
  }
}
