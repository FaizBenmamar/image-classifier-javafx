cat > README.md << 'EOF'
# Image Classifier (JavaFX + TensorFlow)

Desktop app that classifies animal and object photos with Google's Inception (ImageNet) model.

## Requirements
- JDK 21
- Maven 3.9+
- macOS (Apple Silicon), Linux or Windows

## Run
    mvn clean compile
    mvn javafx:run

## Use
1. Click **Load Images** and pick photos.
2. Click an image to select it (red border).
3. Click **Classify**.

## Notes
- The model recognizes the 1,000 ImageNet classes only (for example it has no "rat").
- Model: Inception 5h, Apache 2.0 license (see `src/main/resources/model/LICENSE`).
- Do not edit `imagenet_comp_graph_label_strings.txt`: label order must match the model.
- On Apple Silicon use an ARM (aarch64) JDK; the pom uses TensorFlow Java 1.0.0.
EOF