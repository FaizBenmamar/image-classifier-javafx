package GUI;

import javafx.application.Platform;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.scene.image.*;
import javafx.scene.layout.TilePane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import model.ImagePreprocessor;
import model.ModelClassifier;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;

public class ImageClassifierController {
    @FXML
    private TilePane imageDisplayPane;

    @FXML
    private Button loadButton;
    @FXML
    private Button classifyButton;
    @FXML
    private Button deleteButton;
    @FXML
    private Button resetButton;
    @FXML
    private ProgressIndicator loadingIndicator;

    private final List<VBox> selectedImages = new ArrayList<>();
    private ModelClassifier modelClassifier;

    // Daemon threads: the JVM can exit even if a task is still running
    private final ExecutorService executorService = Executors.newFixedThreadPool(4, r -> {
        Thread t = new Thread(r, "classifier-worker");
        t.setDaemon(true);
        return t;
    });

    @FXML
    public void initialize() {
        loadingIndicator.setVisible(false);

        loadButton.setOnAction(e -> loadImages());
        classifyButton.setOnAction(e -> initiateClassification());
        deleteButton.setOnAction(e -> deleteSelectedImages());
        resetButton.setOnAction(e -> resetClassifications());

        // Load the model once, from the classpath (works from any working directory)
        try {
            modelClassifier = new ModelClassifier(
                    "/model/tensorflow_inception_graph.pb",
                    "/model/imagenet_comp_graph_label_strings.txt"
            );
        } catch (Exception e) {
            e.printStackTrace();
            classifyButton.setDisable(true);
            showError("Model Error", "Could not load the model:\n" + e.getMessage());
        }
    }

    private void loadImages() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.getExtensionFilters().add(
                new FileChooser.ExtensionFilter("Image Files", "*.png", "*.jpg", "*.jpeg"));
        fileChooser.setTitle("Select Images");

        List<File> files = fileChooser.showOpenMultipleDialog(loadButton.getScene().getWindow());
        if (files == null || files.isEmpty()) {
            return;
        }

        for (File file : files) {
            try {
                Image image = new Image(file.toURI().toString());
                if (image.isError()) {
                    throw image.getException();
                }
                ImageView imageView = createSelectableImageView(image);
                Label classificationLabel = new Label("Type: Unknown");
                classificationLabel.setStyle("-fx-font-size: 14px; -fx-text-fill: #333;");

                VBox imageBox = new VBox(imageView, classificationLabel);
                imageBox.setSpacing(5);
                imageBox.setUserData(file); // keep the real File: no fragile URL -> path hacks

                imageDisplayPane.getChildren().add(imageBox);
            } catch (Exception e) {
                showError("Image Load Error", "Error loading image: " + file.getName());
            }
        }
    }

    private ImageView createSelectableImageView(Image image) {
        ImageView imageView = new ImageView(image);
        imageView.setFitWidth(150);
        imageView.setPreserveRatio(true);
        imageView.setOnMouseClicked(event -> handleImageSelection((VBox) imageView.getParent()));
        return imageView;
    }

    private void handleImageSelection(VBox imageBox) {
        if (selectedImages.contains(imageBox)) {
            selectedImages.remove(imageBox);
            imageBox.getStyleClass().remove("image-box-selected");
        } else {
            selectedImages.add(imageBox);
            imageBox.getStyleClass().add("image-box-selected");
        }
    }

    private void deleteSelectedImages() {
        imageDisplayPane.getChildren().removeAll(selectedImages);
        selectedImages.clear();
    }

    private void resetClassifications() {
        for (VBox imageBox : selectedImages) {
            Label classificationLabel = (Label) imageBox.getChildren().get(1);
            classificationLabel.setText("Type: Unknown");
            imageBox.getStyleClass().remove("image-box-selected");
        }
        selectedImages.clear();
    }

    private void showError(String header, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Error");
        alert.setHeaderText(header);
        alert.setContentText(message);
        alert.showAndWait();
    }

    private void initiateClassification() {
        if (selectedImages.isEmpty()) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Nothing selected");
            alert.setHeaderText(null);
            alert.setContentText("Click one or more images to select them first.");
            alert.showAndWait();
            return;
        }

        List<VBox> batch = new ArrayList<>(selectedImages);
        AtomicInteger remaining = new AtomicInteger(batch.size());
        loadingIndicator.setVisible(true);

        for (VBox imageBox : batch) {
            File file = (File) imageBox.getUserData();
            Label classificationLabel = (Label) imageBox.getChildren().get(1);
            classificationLabel.setText("Classifying...");

            executorService.submit(() -> {
                String result;
                try {
                    float[][][][] inputData = ImagePreprocessor.preprocessImage(file.getAbsolutePath());
                    result = modelClassifier.classifyImage(inputData, "input", "softmax2");
                } catch (Exception e) {
                    e.printStackTrace();
                    result = "Error: " + e.getMessage();
                }

                final String text = result;
                Platform.runLater(() -> {
                    classificationLabel.setText(text);
                    // Hide the spinner only when the LAST image is finished
                    if (remaining.decrementAndGet() == 0) {
                        loadingIndicator.setVisible(false);
                    }
                });
            });
        }
    }

    public void shutdownExecutor() {
        executorService.shutdown();
        if (modelClassifier != null) {
            modelClassifier.close();
        }
    }
}
