//router.dart

import 'package:go_router/go_router.dart';
import 'package:document_chat/src/admin_screen.dart';
import 'package:document_chat/src/chat_screen.dart';

final router = GoRouter(
  initialLocation: '/admin',
  routes: [
    GoRoute(
      path: '/admin',
      builder: (context, state) => const AdminScreen(),
    ),
    GoRoute(
      path: '/chat',
      builder: (context, state) => const ChatScreen(),
    ),
  ],
);