package GUI;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.layout.AnchorPane;
import javafx.stage.Stage;
import javafx.scene.image.Image;

import java.io.InputStream;

public class ImageClassifierApp extends Application {

    private ImageClassifierController controller;

    @Override
    public void start(Stage primaryStage) throws Exception {
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/GUI/ImageClassifier.fxml"));
        AnchorPane root = loader.load();
        controller = loader.getController();

        // The icon is optional: don't crash if logo.png is missing
        InputStream logo = getClass().getResourceAsStream("/logo.png");
        if (logo != null) {
            primaryStage.getIcons().add(new Image(logo));
        }

        Scene scene = new Scene(root, 800, 600);
        scene.getStylesheets().add(getClass().getResource("/GUI/styles.css").toExternalForm());

        primaryStage.setTitle("Image Classifier");
        primaryStage.setScene(scene);
        primaryStage.show();
    }

    @Override
    public void stop() {
        if (controller != null) {
            controller.shutdownExecutor();
        }
    }

    public static void main(String[] args) {
        launch(args);
    }
}
