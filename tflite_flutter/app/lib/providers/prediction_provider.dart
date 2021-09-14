import 'package:hooks_riverpod/hooks_riverpod.dart';

final predictionProvider =
    StateNotifierProvider<PredictionController, PredictionModel>(
  (ref) => PredictionController(),
);

class PredictionController extends StateNotifier<PredictionModel> {
  PredictionController() : super(PredictionModel(-1));

  update(recognition) => state = recognition;
}

class PredictionModel {
  PredictionModel(this.digit);
  int digit;
}
