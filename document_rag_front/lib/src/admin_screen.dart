// lib/src/admin_screen.dart
import 'package:document_chat/src/utils/url_helper.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:file_picker/file_picker.dart';
import 'package:go_router/go_router.dart';
import 'package:document_chat/src/providers.dart';
import 'package:document_chat/src/api_service.dart';
import 'package:document_chat/src/widgets/upload_progress_dialog.dart';

class AdminScreen extends ConsumerWidget {
  const AdminScreen({super.key});

  void _showClearAllDialog(BuildContext context, WidgetRef ref) {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('Clear All Data?'),
          content: const Text(
              'This will permanently delete all uploaded documents, their data, '
              'and all chat history. This action cannot be undone.'),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel'),
              onPressed: () => Navigator.of(dialogContext).pop(),
            ),
            TextButton(
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: const Text('Clear All'),
              onPressed: () async {
                Navigator.of(dialogContext).pop();
                try {
                  await ref.read(apiServiceProvider).clearAllDocuments();

                  // Invalidate both providers
                  ref.invalidate(documentsProvider);
                  ref.invalidate(chatProvider); // ADD THIS LINE

                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(
                          content: Text('All data cleared successfully!')),
                    );
                  }
                } on ApiException catch (e) {
                  if (context.mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('Failed to clear: ${e.userMessage}'),
                        backgroundColor: Colors.red,
                      ),
                    );
                  }
                }
              },
            ),
          ],
        );
      },
    );
  }

  void _showDeleteDialog(
      BuildContext context, WidgetRef ref, String docId, String filename) {
    showDialog(
      context: context,
      builder: (BuildContext dialogContext) {
        return AlertDialog(
          title: const Text('Delete Document?'),
          content: Text(
              'Are you sure you want to delete "$filename"? This action cannot be undone.'),
          actions: <Widget>[
            TextButton(
              child: const Text('Cancel'),
              onPressed: () => Navigator.of(dialogContext).pop(),
            ),
            TextButton(
              style: TextButton.styleFrom(foregroundColor: Colors.red),
              child: const Text('Delete'),
              onPressed: () async {
                Navigator.of(dialogContext).pop();

                // Show loading indicator
                showDialog(
                  context: context,
                  barrierDismissible: false,
                  builder: (BuildContext loadingContext) {
                    return const AlertDialog(
                      content: Row(
                        children: [
                          CircularProgressIndicator(),
                          SizedBox(width: 20),
                          Text('Deleting document...'),
                        ],
                      ),
                    );
                  },
                );

                try {
                  await ref.read(apiServiceProvider).deleteDocument(docId);
                  ref.invalidate(documentsProvider);

                  if (context.mounted) {
                    Navigator.of(context).pop(); // Close loading dialog
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                          content: Text('Deleted "$filename" successfully')),
                    );
                  }
                } on ApiException catch (e) {
                  if (context.mounted) {
                    Navigator.of(context).pop(); // Close loading dialog
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        content: Text('Failed to delete: ${e.userMessage}'),
                        backgroundColor: Colors.red,
                      ),
                    );
                  }
                }
              },
            ),
          ],
        );
      },
    );
  }

  Future<void> _handleUpload(BuildContext context, WidgetRef ref) async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['pdf', 'jpg', 'jpeg', 'png'],
    );

    if (result == null || !context.mounted) return;

    final file = result.files.first;

    // Client-side validation
    final allowedExtensions = ['pdf', 'jpg', 'jpeg', 'png'];
    final fileExtension = file.extension?.toLowerCase() ?? '';

    if (!allowedExtensions.contains(fileExtension)) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
              'Unsupported file type: $fileExtension. Allowed: ${allowedExtensions.join(", ")}'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    // Size validation (50MB limit)
    const maxSizeBytes = 50 * 1024 * 1024;
    if (file.size > maxSizeBytes) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('File too large. Maximum size is 50MB.'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      // Start upload and get document ID
      final documentId =
          await ref.read(apiServiceProvider).uploadDocument(file);

      if (!context.mounted) return;

      // Show progress dialog
      final success = await showDialog<bool>(
        context: context,
        barrierDismissible: false,
        builder: (context) => UploadProgressDialog(
          documentId: documentId,
          filename: file.name,
          apiService: ref.read(apiServiceProvider),
        ),
      );

      if (!context.mounted) return;

      if (success == true) {
        ref.invalidate(documentsProvider);
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Document processed successfully!')),
        );
      } else if (success == false) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Document processing failed'),
            backgroundColor: Colors.red,
          ),
        );
      }
    } on ApiException catch (e) {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(e.userMessage),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final documentsAsync = ref.watch(documentsProvider);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Admin - Manage Documents'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Refresh document list',
            onPressed: () => ref.invalidate(documentsProvider),
          ),
          TextButton(
            onPressed: () => context.go('/chat'),
            child:
                const Text('Go to Chat', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
      body: Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Wrap(
                  spacing: 12.0,
                  runSpacing: 12.0,
                  alignment: WrapAlignment.center,
                  children: [
                    ElevatedButton.icon(
                      icon: const Icon(Icons.upload_file),
                      label: const Text('Upload Document'),
                      onPressed: () => _handleUpload(context, ref),
                    ),
                    ElevatedButton.icon(
                      icon: const Icon(Icons.delete_sweep),
                      label: const Text('Clear All Data'),
                      style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.red[700]),
                      onPressed: () => _showClearAllDialog(context, ref),
                    ),
                  ],
                ),
              ),
              const Divider(),
              Expanded(
                child: documentsAsync.when(
                  data: (docs) {
                    if (docs.isEmpty) {
                      return const Center(
                          child: Text('No documents uploaded yet.'));
                    }
                    return ListView.builder(
                      itemCount: docs.length,
                      itemBuilder: (context, index) {
                        final doc = docs[index];
                        return ListTile(
                          leading: const Icon(Icons.picture_as_pdf),
                          title: Text(doc.filename),
                          trailing: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              IconButton(
                                icon: const Icon(Icons.download,
                                    color: Colors.blue),
                                tooltip: 'Download',
                                onPressed: () {
                                  launchDocumentUrl(context, doc.downloadUrl,
                                      documentName: doc.filename);
                                },
                              ),
                              IconButton(
                                icon:
                                    const Icon(Icons.delete, color: Colors.red),
                                tooltip: 'Delete',
                                onPressed: () => _showDeleteDialog(
                                    context, ref, doc.id, doc.filename),
                              ),
                            ],
                          ),
                        );
                      },
                    );
                  },
                  loading: () =>
                      const Center(child: CircularProgressIndicator()),
                  error: (err, stack) =>
                      Center(child: Text('Error loading documents: $err')),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
