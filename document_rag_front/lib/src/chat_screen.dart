// lib/src/chat_screen.dart
import 'package:document_chat/src/models/page_search_result.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:document_chat/src/providers.dart';
import 'package:document_chat/src/models/chat_message.dart';
import 'package:document_chat/src/utils/url_helper.dart';

class ChatScreen extends ConsumerStatefulWidget {
  const ChatScreen({super.key});

  @override
  ChatScreenState createState() => ChatScreenState();
}

class ChatScreenState extends ConsumerState<ChatScreen> {
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _controller.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    ref.listen(chatProvider, (_, __) => _scrollToBottom());
    final messages = ref.watch(chatProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Document Search'),
        actions: [
          IconButton(
            icon: const Icon(Icons.clear_all),
            tooltip: 'Clear history',
            onPressed: () => ref.read(chatProvider.notifier).clearHistory(),
          ),
          TextButton(
            onPressed: () => context.go('/admin'),
            child: const Text('Admin', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 1000),
          child: Column(
            children: [
              Expanded(
                child: ListView.builder(
                  controller: _scrollController,
                  padding: const EdgeInsets.all(16),
                  itemCount: messages.length,
                  itemBuilder: (context, index) => _buildMessage(messages[index]),
                ),
              ),
              _buildChatInput(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMessage(ChatMessage message) {
    final isUser = message.sender == Sender.user;

    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 16),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.85,
        ),
        child: isUser ? _buildUserMessage(message) : _buildAiMessage(message),
      ),
    );
  }

  Widget _buildUserMessage(ChatMessage message) {
    final hasArabic = RegExp(r'[\u0600-\u06FF]').hasMatch(message.text ?? "");

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue[100],
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        message.text ?? "",
        textDirection: hasArabic ? TextDirection.rtl : TextDirection.ltr,
        style: const TextStyle(fontSize: 15),
      ),
    );
  }

  Widget _buildAiMessage(ChatMessage message) {
    if (message.isError) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.red[50],
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.red[300]!),
        ),
        child: Row(
          children: [
            Icon(Icons.error_outline, color: Colors.red[700]),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                message.error!,
                style: TextStyle(color: Colors.red[900]),
              ),
            ),
          ],
        ),
      );
    }

    if (message.isPageResult) {
      return _buildPageResult(message.pageResult!);
    }

    if (message.isText) {
      final hasArabic = RegExp(r'[\u0600-\u06FF]').hasMatch(message.text!);
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.grey[200],
          borderRadius: BorderRadius.circular(12),
        ),
        child: Text(
          message.text!,
          textDirection: hasArabic ? TextDirection.rtl : TextDirection.ltr,
        ),
      );
    }

    return const SizedBox.shrink();
  }

  Widget _buildPageResult(PageSearchResult result) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.blue[50],
              borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: Text(
                        result.documentName,
                        style: const TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Chip(
                      label: Text(
                        result.scorePercentage,
                        style: const TextStyle(fontSize: 12),
                      ),
                      backgroundColor: _getScoreColor(result.score),
                      padding: EdgeInsets.zero,
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Icon(Icons.description, size: 16, color: Colors.grey[600]),
                    const SizedBox(width: 4),
                    Text(
                      'Page ${result.pageNumber}',
                      style: TextStyle(color: Colors.grey[700], fontSize: 14),
                    ),
                    const SizedBox(width: 16),
                    Icon(Icons.star, size: 16, color: Colors.amber[700]),
                    const SizedBox(width: 4),
                    Text(
                      result.relevanceLabel,
                      style: TextStyle(color: Colors.grey[700], fontSize: 14),
                    ),
                  ],
                ),
                if (result.chunkCount > 1) ...[
                  const SizedBox(height: 4),
                  Text(
                    '${result.chunkCount} relevant sections found',
                    style: TextStyle(
                      color: Colors.blue[700],
                      fontSize: 12,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Highlights
          if (result.highlights.isNotEmpty) ...[
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.highlight, size: 16, color: Colors.orange[700]),
                      const SizedBox(width: 8),
                      const Text(
                        'Relevant sections:',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ...result.highlights.map((highlight) => Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.yellow[50],
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.yellow[300]!),
                          ),
                          child: Text(
                            highlight,
                            textDirection: TextDirection.rtl,
                            style: const TextStyle(fontSize: 13, height: 1.5),
                          ),
                        ),
                      )),
                ],
              ),
            ),
          ],

          // Page Image Preview (THUMBNAIL)
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(Icons.image, size: 16, color: Colors.grey[700]),
                    const SizedBox(width: 8),
                    const Text(
                      'Page preview:',
                      style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                GestureDetector(
                  onTap: () => _showFullPageImage(context, result),
                  child: Container(
                    constraints: const BoxConstraints(maxHeight: 400),
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey[300]!),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.network(
                        result.thumbnailUrl, // ✅ THUMBNAIL in list
                        fit: BoxFit.contain,
                        loadingBuilder: (context, child, progress) {
                          if (progress == null) return child;
                          return Center(
                            child: Padding(
                              padding: const EdgeInsets.all(32),
                              child: Column(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const CircularProgressIndicator(),
                                  const SizedBox(height: 16),
                                  Text(
                                    'Loading preview...',
                                    style: TextStyle(color: Colors.grey[600]),
                                  ),
                                ],
                              ),
                            ),
                          );
                        },
                        errorBuilder: (context, error, stack) {
                          return _buildImageFallback(result);
                        },
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),

          // Actions
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.zoom_in, size: 18),
                    label: const Text('View Full Page'),
                    onPressed: () => _showFullPageImage(context, result),
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: OutlinedButton.icon(
                    icon: const Icon(Icons.download, size: 18),
                    label: const Text('Download PDF'),
                    onPressed: () => launchDocumentUrl(
                      context,
                      result.downloadUrl,
                      documentName: result.documentName,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 0.85) return Colors.green[100]!;
    if (score >= 0.70) return Colors.blue[100]!;
    return Colors.orange[100]!;
  }

  // FALLBACK: Image failed in chat list
  Widget _buildImageFallback(PageSearchResult result) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        children: [
          Icon(Icons.article, size: 48, color: Colors.grey[400]),
          const SizedBox(height: 12),
          Text(
            'Preview unavailable',
            style: TextStyle(
              color: Colors.grey[600],
              fontWeight: FontWeight.bold,
            ),
          ),
          if (result.highlights.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              'Text preview:',
              style: TextStyle(color: Colors.grey[500], fontSize: 12),
            ),
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(4),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: Text(
                result.highlights.first,
                textDirection: TextDirection.rtl,
                style: const TextStyle(fontSize: 12),
                maxLines: 3,
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
          const SizedBox(height: 12),
          Text(
            'Tap to view text content',
            style: TextStyle(
              color: Colors.blue[600],
              fontSize: 11,
              fontStyle: FontStyle.italic,
            ),
          ),
        ],
      ),
    );
  }

  // FULLSCREEN: Show original image
  void _showFullPageImage(BuildContext context, PageSearchResult result) {
    String imageUrl;

    if (result.highlightToken != null && result.highlightToken!.isNotEmpty) {
      imageUrl = 'http://100.127.26.110:8000/page-image/by-token/${result.highlightToken}';
    } else {
      // Fallback to original non-highlighted image
      imageUrl = result.imageUrl;
    }

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (context) => Scaffold(
          backgroundColor: Colors.black,
          appBar: AppBar(
            title: Text('${result.documentName} - Page ${result.pageNumber}'),
            backgroundColor: Colors.black87,
            actions: [
              IconButton(
                icon: const Icon(Icons.download),
                tooltip: 'Download PDF',
                onPressed: () => launchDocumentUrl(
                  context,
                  result.downloadUrl,
                  documentName: result.documentName,
                ),
              ),
              IconButton(
                icon: const Icon(Icons.close),
                tooltip: 'Close',
                onPressed: () => Navigator.pop(context),
              ),
            ],
          ),
          body: InteractiveViewer(
            minScale: 0.5,
            maxScale: 4.0,
            child: Center(
              child: Image.network(
                imageUrl,
                fit: BoxFit.contain,
                loadingBuilder: (context, child, progress) {
                  if (progress == null) return child;
                  return const Center(
                    child: CircularProgressIndicator(color: Colors.white),
                  );
                },
                errorBuilder: (context, error, stack) {
                  return _buildFullscreenImageFallback(result);
                },
              ),
            ),
          ),
        ),
      ),
    );
  }

  // FALLBACK: Original image failed in fullscreen
  Widget _buildFullscreenImageFallback(PageSearchResult result) {
    return Container(
      color: Colors.black,
      padding: const EdgeInsets.all(32),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.error_outline, size: 64, color: Colors.red[300]),
            const SizedBox(height: 24),
            const Text(
              'Page image could not be loaded',
              style: TextStyle(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            Text(
              'Showing text content instead:',
              style: TextStyle(color: Colors.grey[400], fontSize: 14),
            ),
            const SizedBox(height: 24),
            Container(
              constraints: const BoxConstraints(maxWidth: 600),
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.grey[900],
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey[700]!),
              ),
              child: SingleChildScrollView(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '${result.documentName} - Page ${result.pageNumber}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 16),
                    ...result.highlights.map((highlight) => Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: Text(
                            highlight,
                            textDirection: TextDirection.rtl,
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 14,
                              height: 1.6,
                            ),
                          ),
                        )),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            OutlinedButton.icon(
              icon: const Icon(Icons.download, color: Colors.white),
              label: const Text(
                'Download PDF',
                style: TextStyle(color: Colors.white),
              ),
              style: OutlinedButton.styleFrom(
                side: const BorderSide(color: Colors.white),
              ),
              onPressed: () => launchDocumentUrl(
                context,
                result.downloadUrl,
                documentName: result.documentName,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildChatInput() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: InputDecoration(
                hintText: 'Type your question...',
                hintStyle: TextStyle(color: Colors.grey[400]),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(24),
                ),
                contentPadding: const EdgeInsets.symmetric(
                  horizontal: 20,
                  vertical: 12,
                ),
              ),
              onSubmitted: (_) => _sendMessage(),
              maxLines: null,
              textInputAction: TextInputAction.send,
            ),
          ),
          const SizedBox(width: 12),
          IconButton(
            icon: const Icon(Icons.send),
            onPressed: _sendMessage,
            style: IconButton.styleFrom(
              backgroundColor: Colors.blue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.all(12),
            ),
          ),
        ],
      ),
    );
  }

  void _sendMessage() {
    if (_controller.text.trim().isNotEmpty) {
      ref.read(chatProvider.notifier).sendMessage(_controller.text.trim());
      _controller.clear();
    }
  }
}
