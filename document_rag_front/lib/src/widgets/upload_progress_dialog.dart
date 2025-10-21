// lib/src/widgets/upload_progress_dialog.dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:document_chat/src/api_service.dart';
import 'package:document_chat/src/models/processing_progress.dart';

class UploadProgressDialog extends StatefulWidget {
  final String documentId;
  final String filename;
  final ApiService apiService;

  const UploadProgressDialog({
    super.key,
    required this.documentId,
    required this.filename,
    required this.apiService,
  });

  @override
  State<UploadProgressDialog> createState() => _UploadProgressDialogState();
}

class _UploadProgressDialogState extends State<UploadProgressDialog> {
  Timer? _pollTimer;
  ProcessingProgress? _progress;
  String? _error;

  @override
  void initState() {
    super.initState();
    _startPolling();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    super.dispose();
  }

  void _startPolling() {
    _pollTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      try {
        final progress = await widget.apiService.getProcessingStatus(widget.documentId);

        if (!mounted) return;

        setState(() {
          _progress = progress;
        });

        if (progress.isComplete) {
          _pollTimer?.cancel();
          Navigator.of(context).pop(true);
        } else if (progress.hasFailed) {
          _pollTimer?.cancel();
          setState(() {
            _error = progress.error ?? 'Processing failed';
          });
        }
      } catch (e) {
        if (!mounted) return;
        setState(() {
          _error = e.toString();
        });
        _pollTimer?.cancel();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Processing Document'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_error != null) ...[
            const Icon(Icons.error, color: Colors.red, size: 48),
            const SizedBox(height: 16),
            Text(
              _error!,
              style: const TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
            ),
          ] else ...[
            CircularProgressIndicator(
              value: _progress != null ? _progress!.progressPercent / 100 : null,
            ),
            const SizedBox(height: 16),
            Text(
              widget.filename,
              style: const TextStyle(fontWeight: FontWeight.bold),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              _progress?.currentStep ?? 'Starting...',
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              '${_progress?.progressPercent ?? 0}%',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
          ],
        ],
      ),
      actions: [
        if (_error != null)
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Close'),
          ),
      ],
    );
  }
}
