// lib/src/models/page_search_result.dart

class PageSearchResult {
  final String documentId;
  final String documentName;
  final int pageNumber;
  final double score;
  final int chunkCount;
  final String imageUrl;
  final String thumbnailUrl;
  final List<String> highlights;
  final String downloadUrl;
  final List<String>? segmentIds; // ADD THIS
  final String? highlightToken; // ADD THIS (if not already there)

  PageSearchResult({
    required this.documentId,
    required this.documentName,
    required this.pageNumber,
    required this.score,
    required this.chunkCount,
    required this.imageUrl,
    required this.thumbnailUrl,
    required this.highlights,
    required this.downloadUrl,
    this.segmentIds, // ADD THIS
    this.highlightToken, // ADD THIS
  });

  factory PageSearchResult.fromJson(Map<String, dynamic> json) {
    return PageSearchResult(
      documentId: json['document_id'] as String,
      documentName: json['document_name'] as String,
      pageNumber: json['page_number'] as int,
      score: (json['score'] as num).toDouble(),
      chunkCount: json['chunk_count'] as int,
      imageUrl: json['image_url'] as String,
      thumbnailUrl: json['thumbnail_url'] as String,
      highlights: List<String>.from(json['highlights'] ?? []),
      downloadUrl: json['download_url'] as String,
      segmentIds: json['segment_ids'] != null // ADD THIS
          ? List<String>.from(json['segment_ids'])
          : null,
      highlightToken: json['highlight_token'] as String?, // ADD THIS
    );
  }

  // Helper getters for UI
  String get scorePercentage => '${(score * 100).toInt()}%';

  String get relevanceLabel {
    if (score >= 0.85) return 'Highly Relevant';
    if (score >= 0.70) return 'Relevant';
    return 'Possibly Relevant';
  }
}
