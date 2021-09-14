import 'package:app/providers/prediction_provider.dart';
import 'package:app/services/classifier.dart';
import 'package:flutter/material.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:image_picker/image_picker.dart';

class DetectPage extends HookConsumerWidget {
  const DetectPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final ImagePicker _picker = ImagePicker();
    final classifier = Classifier();
    final prediction = ref.watch(predictionProvider);
    return Scaffold(
      body: Center(
        child: Text(prediction.digit.toString()),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final XFile? image =
              await _picker.pickImage(source: ImageSource.camera);
          if (image == null) return;
          final prediction = await classifier.classifyImage(image);
          ref.read(predictionProvider.notifier).update(prediction);
        },
        child: const Icon(Icons.camera),
      ),
    );
  }
}
