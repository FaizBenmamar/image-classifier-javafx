package GUI;

/**
 * Plain entry point (does NOT extend Application).
 * Running a class that extends javafx.application.Application directly from the
 * classpath often fails with "JavaFX runtime components are missing".
 * Run THIS class from your IDE / VS Code instead.
 */
public class Launcher {
    public static void main(String[] args) {
        ImageClassifierApp.main(args);
    }
}
