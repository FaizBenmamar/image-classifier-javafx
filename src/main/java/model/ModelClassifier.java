package model;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.GraphDef;
import org.tensorflow.types.TFloat32;

public class ModelClassifier implements AutoCloseable {

    private static final float CONFIDENCE_THRESHOLD = 0.30f;

    private final Graph graph;
    private final Session session;
    private final List<String> labels;
    private final Map<String, String> groups = new HashMap<>();

    public ModelClassifier(String modelResource, String labelsResource) {
        byte[] graphDef = readResource(modelResource);
        graph = new Graph();
        try {
            graph.importGraphDef(GraphDef.parseFrom(graphDef));
        } catch (IOException e) {
            throw new RuntimeException("Invalid model file: " + e.getMessage(), e);
        }
        session = new Session(graph);
        labels = Arrays.asList(new String(readResource(labelsResource), StandardCharsets.UTF_8).split("\\R"));
        initializeGroups();
    }

    private static byte[] readResource(String path) {
        try (InputStream in = ModelClassifier.class.getResourceAsStream(path)) {
            if (in == null) {
                throw new RuntimeException("Resource not found on classpath: " + path);
            }
            return in.readAllBytes();
        } catch (IOException e) {
            throw new RuntimeException("Failed to read " + path + ": " + e.getMessage(), e);
        }
    }

    public String classifyImage(float[][][][] inputData, String inputLayer, String outputLayer) {
        final int h = inputData[0].length;
        final int w = inputData[0][0].length;
        try (TFloat32 input = TFloat32.tensorOf(Shape.of(1, h, w, 3), data -> {
                 for (int y = 0; y < h; y++) {
                     for (int x = 0; x < w; x++) {
                         for (int c = 0; c < 3; c++) {
                             data.setFloat(inputData[0][y][x][c], 0, y, x, c);
                         }
                     }
                 }
             });
             Tensor output = session.runner()
                     .feed(inputLayer, input)
                     .fetch(outputLayer)
                     .run()
                     .get(0)) {

            TFloat32 out = (TFloat32) output;
            int n = (int) out.shape().size(1);
            float[] p = new float[n];
            for (int i = 0; i < n; i++) {
                p[i] = out.getFloat(0, i);
            }

            int best = getBestLabelIndex(p);
            float prob = p[best] * 100;
            String className = best < labels.size() ? labels.get(best).trim() : "class #" + best;
            String group = groups.get(className.toLowerCase());
            String shown = group != null ? group : className;

            if (p[best] < CONFIDENCE_THRESHOLD) {
                return String.format("Not sure - maybe %s (%.0f%%)", shown, prob);
            }
            if (group != null && !group.equalsIgnoreCase(className)) {
                return String.format("Type: %s (%s, %.0f%%)", group, className, prob);
            }
            return String.format("Type: %s (%.0f%%)", className, prob);
        } catch (RuntimeException e) {
            e.printStackTrace();
            return "Error: " + e.getMessage();
        }
    }

    private void add(String group, String... names) {
        for (String name : names) {
            groups.put(name.toLowerCase(), group);
        }
    }

    // Only labels that really exist in the ImageNet list. Add more with add("group", "label1", "label2").
    private void initializeGroups() {
        add("cat", "tabby", "Egyptian cat", "Persian cat", "tiger cat", "Siamese cat");
        add("orca", "killer whale");
        add("whale", "grey whale");
        add("chicken", "hen", "cock");
        add("rodent", "hamster", "guinea pig", "fox squirrel", "marmot", "porcupine", "beaver");
        add("fox", "kit fox", "red fox", "grey fox", "Arctic fox");
        add("wolf", "red wolf", "white wolf", "timber wolf", "coyote");
        add("bear", "brown bear", "American black bear", "ice bear", "sloth bear");
        add("wild dog", "dingo", "dhole", "African hunting dog");
        add("dog", "English setter", "Siberian husky", "Australian terrier", "English springer",
                "malamute", "Great Dane", "Walker hound", "Welsh springer spaniel", "whippet",
                "Scottish deerhound", "Weimaraner", "soft-coated wheaten terrier", "Dandie Dinmont",
                "Old English sheepdog", "otterhound", "bloodhound", "Airedale", "giant schnauzer",
                "black-and-tan coonhound", "dalmatian", "papillon", "Staffordshire bullterrier",
                "Mexican hairless", "Bouvier des Flandres", "miniature poodle", "Cardigan", "malinois",
                "Lhasa", "Yorkshire terrier", "Newfoundland", "Norwegian elkhound", "Rottweiler",
                "Saluki", "schipperke", "Pekinese", "Brabancon griffon", "West Highland white terrier",
                "Sealyham terrier", "Irish wolfhound", "EntleBucher", "French bulldog", "basenji",
                "Bernese mountain dog", "Maltese dog", "Norfolk terrier", "toy terrier", "vizsla",
                "cairn", "groenendael", "clumber", "komondor", "Afghan hound", "Japanese spaniel",
                "borzoi", "toy poodle", "Kerry blue terrier", "Scotch terrier", "Tibetan mastiff",
                "Doberman", "Boston bull", "Greater Swiss Mountain dog", "Appenzeller", "Shih-Tzu",
                "Irish water spaniel", "Pomeranian", "Bedlington terrier", "miniature schnauzer",
                "collie", "golden retriever", "Irish terrier", "affenpinscher", "Border collie",
                "boxer", "silky terrier", "beagle", "Leonberg", "German short-haired pointer",
                "Chesapeake Bay retriever", "bull mastiff", "kuvasz", "pug", "curly-coated retriever",
                "flat-coated retriever", "keeshond", "Eskimo dog", "Brittany spaniel", "standard poodle",
                "Lakeland terrier", "Gordon setter", "standard schnauzer", "Tibetan terrier",
                "wire-haired fox terrier", "basset", "chow", "American Staffordshire terrier",
                "Shetland sheepdog", "Great Pyrenees", "Chihuahua", "Labrador retriever",
                "Saint Bernard", "Samoyed", "bluetick", "redbone", "kelpie", "miniature pinscher",
                "Italian greyhound", "cocker spaniel", "Irish setter", "Sussex spaniel", "Pembroke",
                "Blenheim spaniel", "Rhodesian ridgeback", "Ibizan hound", "English foxhound",
                "briard", "Border terrier", "German shepherd");
    }

    private int getBestLabelIndex(float[] probabilities) {
        int bestIndex = 0;
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > probabilities[bestIndex]) {
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    @Override
    public void close() {
        session.close();
        graph.close();
    }
}