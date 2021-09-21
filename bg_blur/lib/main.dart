import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:tflite/tflite.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as imglib;

List<CameraDescription>? cameras;
Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key}) : super(key: key);
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  Uint8List? _recognitions;
  Uint8List? _picRecognitions;
  double? _imageHeight;
  double? _imageWidth;
  CameraImage? img;
  CameraController? controller;
  bool isBusy = false;

  @override
  void initState() {
    super.initState();
    loadModel();
    initCamera();
  }

  @override
  void dispose() {
    super.dispose();
    controller!.stopImageStream();
    Tflite.close();
  }

  Future loadModel() async {
    Tflite.close();
    try {
      var res = await Tflite.loadModel(
        model: "assets/models/deeplabv3_257_mv_gpu.tflite",
        labels: "assets/models/deeplabv3_257_mv_gpu_person.txt",
        // useGpuDelegate: true,
      );
      print(res);
    } catch (e) {
      print('Failed to load model.: $e');
    }
  }

  initCamera() {
    controller = CameraController(cameras![1], ResolutionPreset.medium);
    controller!.initialize().then(
      (_) {
        if (!mounted) {
          return;
        }
        setState(
          () {
            controller!.startImageStream(
              (image) => {
                if (!isBusy) {isBusy = true, img = image, runModelOnFrame()}
              },
            );
          },
        );
      },
    );
  }

  Future runModelOnImage(File image) async {
    final imglib.Image? capturedImage =
        imglib.decodeImage(await image.readAsBytes());
    final imglib.Image orientedImage = imglib.bakeOrientation(capturedImage!);
    await image.writeAsBytes(imglib.encodeJpg(orientedImage));
    _picRecognitions = await Tflite.runSegmentationOnImage(
      path: image.path,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    setState(() {});
  }

  runModelOnFrame() async {
    _imageWidth = img!.width + 0.0;
    _imageHeight = img!.height + 0.0;
    List<Uint8List> bytesList = [];
    for (var plane in img!.planes) {
      Uint8List planeBytes = plane.bytes;
      File planeBytesFile = File.fromRawPath(planeBytes);
      final imglib.Image? capturedImage = imglib.decodeImage(planeBytes);
      if (capturedImage == null) continue;
      final imglib.Image orientedImage = imglib.bakeOrientation(capturedImage);
      // await planeBytesFile.writeAsBytes(imglib.encodeJpg(orientedImage));
      final planeBytesBake = orientedImage.getBytes(format: imglib.Format.rgb);
      bytesList.add(planeBytesBake);
    }
    print(bytesList);
    // if (bytesList.isEmpty) return;

    _recognitions = await Tflite.runSegmentationOnFrame(
      // bytesList: bytesList,
      bytesList: img!.planes.map((plane) {
        return plane.bytes;
      }).toList(),
      imageHeight: img!.height,
      imageWidth: img!.width,
      imageMean: 127.5,
      imageStd: 127.5,
      outputType: 'png',
      asynch: true,
      rotation: 90,
    );
    print(_recognitions!.length);
    isBusy = false;
    setState(
      () {
        img;
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    List<Widget> stackChildren = [];
    stackChildren.add(
      Positioned(
        top: 0.0,
        left: 0.0,
        width: size.width,
        height: size.height,
        child: Container(
          height: size.height,
          child: (!controller!.value.isInitialized)
              ? Container()
              : CameraPreview(controller!),
          constraints: BoxConstraints(
            maxHeight: 550,
            maxWidth: size.width,
          ),
        ),
      ),
    );

    if (_recognitions != null) {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          // width: size.width,
          height: size.height,
          child: Opacity(
            opacity: 0.4,
            child: Image.memory(
              _recognitions!,
              fit: BoxFit.fill,
            ),
          ),
        ),
      );
    }

    if (_picRecognitions != null) {
      stackChildren.add(
        Positioned(
          top: 0.0,
          left: 0.0,
          width: size.width,
          height: size.height,
          child: Opacity(
            opacity: 1,
            child: Image.memory(
              _picRecognitions!,
              fit: BoxFit.fill,
            ),
          ),
        ),
      );
    }

    return SafeArea(
      child: Scaffold(
        body: Stack(
          children: stackChildren,
        ),
        floatingActionButton: FloatingActionButton(
          child: const Icon(
            Icons.camera,
          ),
          onPressed: () async {
            await controller!.stopImageStream();
            final pic = await controller!.takePicture();
            await runModelOnImage(File(pic.path));
          },
        ),
      ),
    );
  }
}
