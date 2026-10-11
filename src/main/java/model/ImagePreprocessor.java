package model;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import org.imgscalr.Scalr;

public class ImagePreprocessor {

    private static final int SIZE = 224;
    private static final float MEAN = 117f;
    private static final float SCALE = 1f;

    public static float[][][][] preprocessImage(String imagePath) throws IOException {
        File imageFile = new File(imagePath);
        if (!imageFile.exists() || !imageFile.canRead()) {
            throw new IOException("Cannot read image file: " + imagePath);
        }

        BufferedImage img = ImageIO.read(imageFile);
        if (img == null) {
            throw new IOException("Failed to load image file (unsupported format?).");
        }

        // Scale so the SHORT side is 224, keeping the aspect ratio, then center-crop 224x224
        BufferedImage scaled = (img.getWidth() < img.getHeight())
                ? Scalr.resize(img, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_TO_WIDTH, SIZE, Scalr.OP_ANTIALIAS)
                : Scalr.resize(img, Scalr.Method.ULTRA_QUALITY, Scalr.Mode.FIT_TO_HEIGHT, SIZE, Scalr.OP_ANTIALIAS);
        int x0 = (scaled.getWidth() - SIZE) / 2;
        int y0 = (scaled.getHeight() - SIZE) / 2;
        BufferedImage resized = Scalr.crop(scaled, x0, y0, SIZE, SIZE);

        float[][][][] input = new float[1][SIZE][SIZE][3];
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int rgb = resized.getRGB(x, y);
                input[0][y][x][0] = (((rgb >> 16) & 0xFF) - MEAN) / SCALE;
                input[0][y][x][1] = (((rgb >> 8) & 0xFF) - MEAN) / SCALE;
                input[0][y][x][2] = ((rgb & 0xFF) - MEAN) / SCALE;
            }
        }
        return input;
    }
}