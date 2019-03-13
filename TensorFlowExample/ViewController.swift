//
//  ViewController.swift
//  TensorFlowExample
//
//  Created by Rogier van den Brink on 12/03/2019.
//  Copyright © 2019 Incentro. All rights reserved.
//

import UIKit
import Firebase
import FirebaseMLCommon
import AVFoundation

class ViewController: UIViewController {

    @IBOutlet weak var classificationLabel: UILabel!
    @IBOutlet weak var imageView: UIImageView!
    var session: AVCaptureSession?
    
    override func viewDidLoad() {
        startLiveVideo()
    }
    
    private func startLiveVideo() {
        let session = AVCaptureSession()
        self.session = session
        
        session.sessionPreset = AVCaptureSession.Preset.photo
        let captureDevice = AVCaptureDevice.default(for: AVMediaType.video)
        
        let deviceInput = try! AVCaptureDeviceInput(device: captureDevice!)
        let deviceOutput = AVCaptureVideoDataOutput()
        deviceOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
        deviceOutput.setSampleBufferDelegate(self, queue: DispatchQueue.global(qos: DispatchQoS.QoSClass.default))
        session.addInput(deviceInput)
        session.addOutput(deviceOutput)
        
        let imageLayer = AVCaptureVideoPreviewLayer(session: session)
        imageLayer.frame = imageView.bounds
        imageLayer.videoGravity = .resizeAspectFill
        imageView.layer.addSublayer(imageLayer)
        
        session.startRunning()
    }

    func interpretImage(imageToInterpret: CGImage) {
        
        // MARK: Load Local Model
        guard let modelPath = Bundle.main.path(forResource: "optimized_graph", ofType: "tflite")
            else {
                // Invalid model path
                print("invalid model path")
                return
        }
        
        let localModelSource = LocalModelSource(
            name: "optimized_graph",
            path: modelPath)
        let conditions = ModelDownloadConditions(isWiFiRequired: true, canDownloadInBackground: true)
        
        // MARK: Load Cloud Model
        let cloudModelSource = CloudModelSource(
            name: "optimized_graph",
            enableModelUpdates: true,
            initialConditions: conditions,
            updateConditions: conditions
        )
        let registrationSuccessful = ModelManager.modelManager().register(cloudModelSource)

        // MARK: Create an interpreter
        let options = ModelOptions(
            cloudModelName: "optimized_graph",
            localModelName: nil)
        let interpreter = ModelInterpreter.modelInterpreter(options: options)
        
        // MARK: Specify the model's input and output
        let ioOptions = ModelInputOutputOptions()
        do {
            try ioOptions.setInputFormat(index: 0, type: .float32, dimensions: [1, 224, 224, 3])
            try ioOptions.setOutputFormat(index: 0, type: .float32, dimensions: [1, 5])
        } catch let error as NSError {
            print("Failed to set input or output format with error: \(error.localizedDescription)")
        }
        
        let image: CGImage = imageToInterpret
        guard let context = CGContext(
            data: nil,
            width: image.width, height: image.height,
            bitsPerComponent: 8, bytesPerRow: image.width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
            ) else {
                return
        }
        
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        guard let imageData = context.data else { return }
        
        let inputs = ModelInputs()
        var inputData = Data()
        do {
            for row in 0 ..< 224 {
                for col in 0 ..< 224 {
                    let offset = 4 * (col * context.width + row)
                    // (Ignore offset 0, the unused alpha channel)
                    let red = imageData.load(fromByteOffset: offset+1, as: UInt8.self)
                    let green = imageData.load(fromByteOffset: offset+2, as: UInt8.self)
                    let blue = imageData.load(fromByteOffset: offset+3, as: UInt8.self)
                    
                    // Normalize channel values to [0.0, 1.0]. This requirement varies
                    // by model. For example, some models might require values to be
                    // normalized to the range [-1.0, 1.0] instead, and others might
                    // require fixed-point values or the original bytes.
                    var normalizedRed = Float32(red) / 255.0
                    var normalizedGreen = Float32(green) / 255.0
                    var normalizedBlue = Float32(blue) / 255.0
                    
                    // Append normalized values to Data object in RGB order.
                    let elementSize = MemoryLayout.size(ofValue: normalizedRed)
                    var bytes = [UInt8](repeating: 0, count: elementSize)
                    memcpy(&bytes, &normalizedRed, elementSize)
                    inputData.append(&bytes, count: elementSize)
                    memcpy(&bytes, &normalizedGreen, elementSize)
                    inputData.append(&bytes, count: elementSize)
                    memcpy(&bytes, &normalizedBlue, elementSize)
                    inputData.append(&bytes, count: elementSize)
                }
            }
            try inputs.addInput(inputData)
        } catch let error {
            print("Failed to add input: \(error)")
        }

        // MARK: Run interpreter
        interpreter.run(inputs: inputs, options: ioOptions) { outputs, error in
            guard error == nil, let outputs = outputs else { return }
            // Process outputs
            // Get first and only output of inference with a batch size of 1
            let output = try? outputs.output(index: 0) as? [[NSNumber]]
            
            guard let probabilities = output??[0] else { return }
            
            guard let labelPath = Bundle.main.path(forResource: "retrained_labels", ofType: "txt") else { return }
            let fileContents = try? String(contentsOfFile: labelPath)
            guard let labels = fileContents?.components(separatedBy: "\n") else { return }
            
            
            let sortedProbs = probabilities.sorted(by: {$0.doubleValue > $1.doubleValue})
            if let firstProb = sortedProbs.first,
                let index = probabilities.firstIndex(of: firstProb) {
            
                self.classificationLabel.text = "\(labels[index]): \(sortedProbs.first!)"
            }
        }
    }
    
    func convertCIImageToCGImage(inputImage: CIImage) -> CGImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(inputImage, from: inputImage.extent) {
            return cgImage
        }
        return nil
    }
}

extension ViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        if let cgImage = self.convertCIImageToCGImage(inputImage: ciImage) {
        
            self.interpretImage(imageToInterpret: cgImage)
        }
    }
}
