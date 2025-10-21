// lib/src/utils/url_helper.dart
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

Future<void> launchDocumentUrl(
  BuildContext context,
  String url, {
  String? documentName,
}) async {
  final uri = Uri.parse(url);

  if (!await launchUrl(uri, mode: LaunchMode.externalApplication)) {
    if (!context.mounted) return;

    final message = documentName != null
        ? 'Document "$documentName" not available (may have been deleted)'
        : 'Could not open document (may have been deleted)';

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.orange,
      ),
    );
  }
}
