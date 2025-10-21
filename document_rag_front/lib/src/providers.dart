// lib/src/providers.dart
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:document_chat/src/api_service.dart';
import 'package:document_chat/src/models/document.dart';
import 'package:document_chat/src/models/chat_message.dart';
import 'package:document_chat/src/models/page_search_result.dart';

final apiServiceProvider = Provider((ref) => ApiService());

final documentsProvider = FutureProvider<List<Document>>((ref) async {
  return await ref.watch(apiServiceProvider).listDocuments();
});

final chatProvider =
    StateNotifierProvider<ChatNotifier, List<ChatMessage>>((ref) {
  return ChatNotifier(ref.watch(apiServiceProvider));
});

class ChatNotifier extends StateNotifier<List<ChatMessage>> {
  final ApiService _apiService;

  ChatNotifier(this._apiService) : super([]) {
    _loadHistory();
  }

  Future<void> _loadHistory() async {
    try {
      final history = await _apiService.getChatHistory();
      final messages = <ChatMessage>[];

      for (final item in history) {
        final sender = item['sender'] as String;
        final content = item['content'] as String;

        if (sender == 'user') {
          messages.add(ChatMessage(sender: Sender.user, text: content));
        } else if (sender == 'ai_result') {
          try {
            final resultData = json.decode(content);
            final pageResult = PageSearchResult(
              documentId: resultData['document_id'],
              documentName: resultData['document_name'],
              pageNumber: resultData['page_number'],
              score: (resultData['score'] ?? 0.0).toDouble(),
              chunkCount: resultData['chunk_count'] ?? 1,
              imageUrl: resultData['image_url'] ?? '',
              thumbnailUrl: resultData['thumbnail_url'] ?? '',
              highlights: List<String>.from(resultData['highlights'] ?? []),
              downloadUrl: resultData['download_url'],
            );
            messages
                .add(ChatMessage(sender: Sender.ai, pageResult: pageResult));
          } catch (e) {
            messages.add(ChatMessage(sender: Sender.ai, text: content));
          }
        } else if (sender == 'ai') {
          messages.add(ChatMessage(sender: Sender.ai, text: content));
        }
      }

      state = messages;
    } catch (e) {
      debugPrint('Failed to load chat history: $e');
    }
  }

  Future<void> sendMessage(String message) async {
    state = [...state, ChatMessage(sender: Sender.user, text: message)];

    try {
      final results = await _apiService.searchPages(message);

      if (results.isEmpty) {
        state = [
          ...state,
          ChatMessage(
            sender: Sender.ai,
            error: 'No relevant pages found.\n\n'
                'Try:\n'
                '• Different keywords\n'
                '• Simpler question\n'
                '• Check document is uploaded',
          )
        ];
      } else {
        final aiMessages = results
            .map((result) => ChatMessage(sender: Sender.ai, pageResult: result))
            .toList();

        const maxMessages = 100;
        final allMessages = [...state, ...aiMessages];

        state = allMessages.length > maxMessages
            ? allMessages.sublist(allMessages.length - maxMessages)
            : allMessages;
      }
    } on ApiException catch (e) {
      state = [
        ...state,
        ChatMessage(sender: Sender.ai, error: 'Error: ${e.userMessage}')
      ];
    }
  }

  Future<void> clearHistory() async {
    await _apiService.clearChatHistory();
    state = [];
  }
}
