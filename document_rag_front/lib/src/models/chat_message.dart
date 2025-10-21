// lib/src/models/chat_message.dart
import 'package:document_chat/src/models/page_search_result.dart';

enum Sender { user, ai }

class ChatMessage {
  final Sender sender;
  final String? text;
  final PageSearchResult? pageResult; // ✅ Changed from searchResult
  final String? error;

  ChatMessage({
    required this.sender,
    this.text,
    this.pageResult, // ✅ Changed parameter name
    this.error,
  });

  // Helper getters
  bool get isError => error != null;
  bool get isText => text != null && pageResult == null && error == null;
  bool get isPageResult => pageResult != null;
}
