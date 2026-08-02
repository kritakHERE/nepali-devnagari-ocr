8\. Critical Analysis

When designing an optical character recognition (OCR) system for handwritten Nepali forms, it is crucial to analyze the current OCR technologies and determine the techniques that can meet the project's needs. The three popular methods used are analyzed: Tesseract OCR, EasyOCR and the Convolutional Recurrent Neural Network (CRNN) suggested by Shi et al. (2016). The solutions were chosen as they span various generations of OCR technology, from legacy OCR engine to the new generation of OCR models based on sequence recognition using deep learning.

Each solution has its own pros and cons on the use of handwritten Devanagari text in structured forms. This project analyzes their architectures, advantages and disadvantages and determines which parts of these architectures are suitable for the proposed system. To this, the final solution incorporates concepts from these current solutions in addition to some other techniques such as template-based form alignments, region of interest extraction, transfer learning from Hindi to Nepali handwriting and lightweight NLP-based post-processing to enhance the image recognition rate.

8.1 Tesseract OCR

Tesseract OCR is a widely used open source optical character recognition (OCR) engine and has been used to a great extent in the field of printed document digitisation. Tesseract was developed by Hewlett-Packard and later maintained by Google, and is available for over 100 languages, including Nepali (Smith, 2007\) by its official nep language package. It is easily available, open source, and being used as the common ground for comparison of OCR systems, because of its maturity, ease of deployment, and availability.

Starting from version 4.0, Tesseract has integrated a Long Short-Term Memory (LSTM) neural network, along with language model based post-processing to enhance the accuracy of text recognition. The LSTM network is used to identify sequences of characters and the language model is used to narrow down the prediction by taking into account the probable patterns of words. This architecture has allowed Tesseract to obtain a character recognition accuracy of more than 95% for good quality printed documents. However, performance drops precipitously for handwritten text, and is usually between 40% and 70% accurate, depending on the quality of the handwriting and the properties of the dataset (Smith, 2007).

Tesseract has a number of merits for this project. It is stable, has good documentation, is available in Nepali and works great without the need for dedicated GPU hardware. These features enable it to be used for a variety of document digitisation projects that deal with printed text.

Tesseract has some of these advantages, but it is not well suited for handwritten Nepali forms. It is mainly trained on printed documents and often does not recognize devanagari characters such as vowel modifiers (matras) and complex character combinations of devanagari characters. In addition, Tesseract does not know the structure of document templates; it can not automatically recognize or retrieve information from the predefined form fields. This restriction is a blocker for structured handwritten forms.

Due to these constraints, we had to consider our proposed system for the work. Instead of using OCR engine to identify the individual fields, we first align the forms scanned by ORB feature matching and homography estimation, then crop the region of interest (ROI). The OCR model is able to get clearer and consistent inputs by isolating each predefined field before recognition, which results in a better OCR accuracy for handwritten Nepali forms.

In a general sense, Tesseract offers a good starting point for printed document OCR, however it is not suitable to use as a stand-alone implementation in this project as it fails to perform well in handwritten Devnagari form recognition.

8.2 EasyOCR

With over 80 languages supported, EasyOCR is an open-source optical character recognition framework developed by JaidedAI which includes Nepali. EasyOCR is not just another OCR engine but is based on deep learning approaches, and has a two-stage pipeline: “text detection” and “text recognition”. It is easy to integrate and supports multiple languages, and this has made it a favorite for contemporary OCR applications for natural images and scanned documents.

Initially, the Character Region Awareness for Text Detection (CRAFT) model with VGG16 is applied to locate regions with text in an image. Detected text regions are cropped and fed into a Convolutional Recurrent Neural Network (CRNN) that uses Bidirectional Long Short-Term Memory (BiLSTM) layers and Connectionist Temporal Classification (CTC) to decode the identified character sequence. Unlike explicit character segmentation, this detect then recognise architecture allows EasyOCR to process text without the need for any segmentation. (Baek et al., 2019\)

It has been shown to be effective in printed Latin text with over 90% accuracy on printed high quality text in published evaluations. Community evaluations also report 60-80% recognition on printed Nepali documents, but performance starts to drop significantly in the case of handwritten documents, where handwritten training data is limited.

EasyOCR has a number of key benefits for this project. It supports Nepali language natively, implements a powerful two-stage recognition pipeline and employs a CRNN with CTC decoding architecture which is similar to our proposed recognition strategy. The performance of deep learning methods for multilingual OCR applications is shown by these features.

But, there are also some drawbacks of using EasyOCR for Nepali handwritten forms. The pre-trained models are mainly trained for the printed text and are not as suitable to handwritten devanagari characters. Besides, form borders and table lines are also detected as text regions by CRAFT text detector, which decreases the detection accuracy on structured forms. Fine tuning of the entire EasyOCR system takes a lot of skill and computational power, which makes it less feasible for a specialized handwritten Nepali OCR system.

In this project, rather than deploying EasyOCR itself, the underlying CRNN and CTC recognition principle is used, but with a domain-specific model which has been pre-trained on Hindi text and fine-tuned on Nepali text. This approach maintains the merits of the EasyOCR's recognition structure and enhances its applicability for handwritten Nepali forms.

In general, EasyOCR is a good reference to the architecture of this project. While it is not an adequately specialized system for handwritten Nepali form recognition for general use, it is a deep learning framework that gave a lot of inspiration to design the recognition pipeline.

8.3 Convolutional Recurrent Neural Network (CRNN)

Shi, Bai and Yao (2016) proposed the Convolutional Recurrent Neural Network (CRNN), which is one of the most significant deep learning models for image-based sequence recognition. It is intended to identify text directly from an image, without having to segment the individual characters first. With this end-to-end method, the CRNN architecture has become widely-used for OCR systems, particularly scripts with complex character structure.

There are four main components of the original CRNN architecture. The first step is a Convolutional Neural Network (CNN) that extracts visual features from the input image. These maps are then mapped to a sequence (column-wise) in a map-to-sequence operation with each column of a feature map being a time step. This generates a sequence that is then fed into Bidirectional Long Short-Term Memory (BiLSTM) layers, which process information in both the forward and backward directions. Lastly, the output text sequence is produced by a Connectionist Temporal Classification (CTC) layer without any need to label each character in the text sequence during training (Shi et al., 2016).

The original CRNN has been evaluated on the printed English text dataset of IIIT5K with an accuracy of 97.8% demonstrating its usefulness for scene text recognition. Later work using the CRNN variants on printed Devanagari text reported around 85-90% recognition rates which shows potential for the architecture to be adapted to Indic scripts with appropriate training data.

There are a few benefits of CRNN in handwritten Nepali OCR. The model is able to recognize complete character sequences, not only single characters which eliminates the need for manual character segmentation, especially for connected hand writing. The BiLSTM layers learn the contextual relationships between the neighboring characters that are beneficial in identifying conjunct consonants and other sequential patterns prevalent in Devanagari script. Moreover, the end to end training process enables the optimization of the whole network, which consequently leads to better recognition performance.

The original CRNN model has limitations for this application, too. Initially created with data from the English language, it does not directly consider the structural complexity of Devanagari writing. Vowel modifiers (matras), conjunct consonants and vertically positioned characters add to the spatial challenges posed by Nepali characters. Besides that, the performance of CRNN is highly sensitive to the amount and quality of the labelled training data, which is still scarce for handwritten Nepali documents.

In contrast to the previous solutions, CRNN was analysed and then directly applied and extended in our proposed system. A pre-trained VGG16 backbone was used in place of the original shallow CNN feature extractor to enhance the feature representation. To further improve the recognition model, the network was first trained on a larger handwritten Hindi database followed by fine tuning on handwritten Nepali database using differential learning rates. This method helped the model capitalize on the similarities between the two scripts and to be flexible in the features of the Nepali handwriting.

In general, the core part of a proposed system is CRNN as recognition engine. The project starts from an existing OCR system, incorporates transfer learning and a more powerful feature extractor, and tackles the real-world problems of handwritten Nepali form recognition.

8.4 Comparison of Existing OCR Solutions

| Criteria | Tesseract OCR | EasyOCR | CRNN(Shit et al., 2016\) |
| :---- | :---- | :---- | :---- |
| Architecture | LSTM- based OCR with language model post-processing | CRAFT text detector (VGG16-based) \+ CRNN(BiLSTM \+ CTC) | CNN feature extractor \-.\> Map-to-Sequence \-\> BiLSTM \-\> CTC |
| Primary Training Data | Printed documents | Printed multilingual text | Printed scene text(original model) |
| Handwritten Text Support | Limited | Moderate | Strong when trained on suitable handwritten datasets |
| Nepali language support | Yes | Yes |  Requires custom training or fine-tuning |
| Form Field Extraction | Not supported | Limited. Text detector may misidentify form boundaries | Not provided.external preprocessing is required |
| Best Reported Devanagari Performance | Moderate on handwritten text | Moderate on printed Nepali and lower on handwritten | 85 to 90 % on printed Devanagari using CRNN variants |
| Deployment Feasibility in Nepal | High(lightweight,CPU-friendly) | Moderate | Moderate to HIgh(depends on trained model and the hardware available) |
| Open Source | Yes | Yes | Yes |
| Key Contribution to our project | Motivated the template alignment and ROI extraction pipeline | Confirmed the effectiveness of the CRNN \+ CTC recognition strategy | Forms foundation of recognition model, extended with VGG16 and HIndi-to-Nepali transfer learning |

8.5 Synthesis

The study on Tesseract OCR, EasyOCR and CRNN reveals that none of the solution meets all the requirements of handwritten Nepali form recognition. Both systems have its own special merits and individual weaknesses that have to be overcome when designing an OCR system for structured handwritten documents.

Assessment of Tesseract OCR revealed that even if it can achieve good accuracy on printed Nepali text and supports Nepali language, it does not have an understanding of document structure and also is not good at hand written Devanagari characters. These restrictions led to the creation of a template-based preprocessing stage for our system. The proposed pipeline is designed to align and isolate each field of handwritten text prior to being fed to the text recognition component using the ORB feature matching, homography estimation and region-of-interest (ROI) cropping.

Likewise, EasyOCR achieved high performance by integrating the deep learning-based text detection technology with CRNN and CTC sequence recognition. Its pre-trained models were not very specialised for handwritten Nepali forms, but recognition framework showed that sequence based OCR is an appropriate approach to multilingual form recognition. This had an impact on the overall design of our recognition pipeline, and enabled us to use a pre-trained model that has been optimized for handwritten Devanagari.

The proposed system was based on the CRNN architecture, proposed by Shi et al. (2016), as it was the most solid foundation. The project does not simply take the original model and apply it as is, but rather modifies it by swapping out the feature extractor with a pre-trained VGG16 network and adds a two-stage transfer learning approach. The model is initially trained on Hindi handwriting data and then fine tuned using Nepali handwritings with differential learning rates, so that the knowledge can be transferred between two similar scripts and, at the same time, the model can learn the writing characteristics specific to the Nepali handwriting.

This project adds a number of extra components to the recognition model that are not found in the systems analyzed. These comprise a designed alignment pipeline for structured forms, an NLP post processing stage containing Unicode normalisation and vocabulary based fuzzy matching and a detailed evaluation using real handwritten Nepali forms. All of these improvements tackle real-world issues in document processing and bridge the gap between lab-based OCR accuracy and its application in Nepalese administrative contexts.

Overall, the proposed system is not directly applicable to any particular solution already in place. Otherwise, it integrates the best ideas and methods of past researches and suggests some specific improvements, which tackle the particular challenges of handwritten Nepali form recognition.

## **References**

Baek, Y., Lee, B., Han, D., Yun, S., & Lee, H. (2019). *Character region awareness for text detection*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 9365–9374.

Shi, B., Bai, X., & Yao, C. (2016). *An end-to-end trainable neural network for image-based sequence recognition and its application to scene text recognition*. IEEE Transactions on Pattern Analysis and Machine Intelligence, 39(11), 2298–2304.

Smith, R. (2007). *An overview of the Tesseract OCR engine*. Proceedings of the Ninth International Conference on Document Analysis and Recognition, 629–633.

