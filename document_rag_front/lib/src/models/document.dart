// lib/src/models/document.dart
class Document {
  final String id;
  final String filename;
  final String downloadUrl; // CHANGED

  Document({
    required this.id,
    required this.filename,
    required this.downloadUrl, // CHANGED
  });

  factory Document.fromJson(Map<String, dynamic> json) {
    return Document(
      id: json['id'] as String,
      filename: json['filename'] as String,
      downloadUrl: json['download_url'] as String, // CHANGED
    );
  }
}
