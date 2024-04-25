package huplay.demo.config;

import java.io.*;
import java.util.*;

import static huplay.demo.App.OUT;

/**
 * Holder of the configuration stored in the model.properties file
 */
public class Config
{
    private final Arguments arguments;

    private final String name;
    private final String transformerType;

    private final String tokenizer;
    private final String tokenizerConfig;
    private final int tokenCount;
    private final int endOfTextToken;
    private final int maxLength;

    private final int hiddenSize;
    private final int feedForwardSize;
    private final int decoderCount;
    private final int headCount;
    private final float epsilon;

    private final String parameterUrl;
    private final List<String> parameterFiles;
    private final String transformerParameterFormat;
    private final String decoderParameterFormat;
    private final Map<String, String> transformerParameterOverrides;
    private final Map<String, String> decoderParameterOverrides;

    private ParameterReader reader;

    public Config(Arguments arguments) throws Exception
    {
        this.arguments = arguments;

        // Read all properties from the model.properties file
        Map<String, String> properties = readProperties(getConfigPath() + "model.properties");
        this.name = getProperty(properties, "name");
        this.transformerType = getProperty(properties, "transformer.type");
        this.tokenizer = getProperty(properties, "tokenizer");
        this.tokenizerConfig = getProperty(properties, "tokenizer.config");
        this.tokenCount = getIntProperty(properties, "token.count");
        this.endOfTextToken = getIntProperty(properties, "end.of.text.token");
        this.maxLength = getIntProperty(properties, "max.length");

        this.hiddenSize = getIntProperty(properties, "hidden.size");
        this.feedForwardSize = getIntProperty(properties, "feedforward.size");
        this.decoderCount = getIntProperty(properties, "decoder.count");
        this.headCount = getIntProperty(properties, "attention.head.count");
        this.epsilon = getFloatProperty(properties, "epsilon");

        this.parameterUrl = getProperty(properties, "parameter.url", true);
        this.parameterFiles = getParameterFiles(properties);
        this.transformerParameterFormat = properties.get("transformer.parameter.format");
        this.decoderParameterFormat = properties.get("decoder.parameter.format");
        this.transformerParameterOverrides = getParameterOverrides(properties, "transformer.parameter.overrides");
        this.decoderParameterOverrides = getParameterOverrides(properties, "decoder.parameter.overrides");

        // Print settings
        OUT.println("Model: " + name);
        OUT.println("Path: " + getModelPath());
        //OUT.print("Number of parameters: " + Math.round(getParameterSize() / 1000000d) + " M");
        OUT.println("Hidden size: " + hiddenSize +
                ", decoders: " + decoderCount +
                ", heads: " + headCount +
                ", head size: " + getHeadSize());

        long cost = getCalculationCost(50, 100);
        OUT.println("Inference calculation cost: " + cost / 1_000_000_000 + " GigaFLOPS");

        if (arguments.isCalculationOnly()) System.exit(0);

        OUT.println("Maximum length of generated text: " + arguments.getLengthLimit());
        OUT.println("Output is selected from the best " + arguments.getTopK() + " tokens (topK)");
    }

    private List<String> getParameterFiles(Map<String, String> properties) throws Exception
    {
        return splitProperty(getProperty(properties, "parameter.files"));
    }

    private Map<String, String> getParameterOverrides(Map<String, String> properties, String name) throws Exception
    {
        String property = getProperty(properties, name, true);

        if (property == null || property.isEmpty())
        {
            return null;
        }

        List<String> overrides = splitProperty(property);

        Map<String, String> result = new HashMap<>();

        for (String override : overrides)
        {
            String[] parts = override.split(":");
            result.put(parts[0].trim(), parts[1].trim());
        }

        return result;
    }

    private List<String> splitProperty(String property)
    {
        List<String> result = new ArrayList<>();

        String[] values = property.split(",");

        for (String value : values)
        {
            result.add(value.trim());
        }

        return result;
    }

    public static Map<String, String> readProperties(String fileName) throws Exception
    {
        Map<String, String> properties = new HashMap<>();

        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNextLine())
            {
                String line = scanner.nextLine();
                if (line != null && !line.trim().isEmpty() && !line.startsWith("#"))
                {
                    String[] parts = line.split("=");
                    if (parts.length == 1)
                    {
                        properties.put(parts[0].trim(), "");
                    }
                    else if (parts.length > 1)
                    {
                        properties.put(parts[0].trim(), parts[1].trim());

                        if (parts.length > 2)
                        {
                            OUT.println("\nWARNING: Additional \"=\" character in value: (" + fileName + "): " + line);
                        }
                    }
                }
            }
        }
        catch (IOException e)
        {
            throw new Exception("Cannot read model.properties file: " + fileName);
        }

        return properties;
    }

    public long getCalculationCost(int inputLength, int outputLength)
    {
        long normCost = 4L * hiddenSize; // TODO: Sqrt averagediff + epsion is missing, but not much

        long layer1Cost = dotProductCost(hiddenSize) * feedForwardSize // weights
                + feedForwardSize // biases
                + 8L * feedForwardSize; // gelu

        long layer2Cost = dotProductCost(feedForwardSize) * hiddenSize + hiddenSize;

        long decoderCost = 2 * normCost // norm cost
                + getAverageAttentionCost(inputLength + outputLength) // Attention mechanism
                + layer1Cost + layer2Cost // Feed forward block
                + hiddenSize * 2L; // Residual connections

        long determineOutputCost = dotProductCost(hiddenSize) * tokenCount + softmaxCost(40);

        // TODO: Order and weighted random pick is missing

        return hiddenSize * (inputLength + outputLength)
                + decoderCost * decoderCount * inputLength + outputLength
                + normCost * (outputLength + 1) // Final normalization
                + determineOutputCost * (outputLength + 1);
    }

    private long dotProductCost(long size)
    {
        return 2 * size - 1;
    }

    private long softmaxCost(int size)
    {
        return 4L * size;

        // Exp calculation was added as multiplication
    }

    private long getAverageAttentionCost(int length)
    {
        long QKVCost = dotProductCost(hiddenSize) * 3 * hiddenSize;

        long attentionCost = dotProductCost(hiddenSize)
                + 1L // Attention dividend
                + softmaxCost(hiddenSize)
                + 2L * hiddenSize;

        return QKVCost + attentionCost * length / 2;
    }

    /*public long getParameterSize()
    {
        long wteSize = (long) tokenCount * hiddenSize;

        long wpeSize = positionEncoder.equals("LEARNED") ? (long) maxLength * hiddenSize : 0;

        long finalNormSize = (long) hiddenSize * 2;

        return wteSize + wpeSize + (getDecoderParameterSize() * decoderCount) + finalNormSize;
    }*/

    private long getDecoderParameterSize()
    {
        long qkvSize = ((long) hiddenSize * hiddenSize + hiddenSize) * 3;
        long projSize = (long) hiddenSize * hiddenSize + hiddenSize;
        long normSize = (long) hiddenSize * 4;
        long layer1Size = ((long) hiddenSize * feedForwardSize + feedForwardSize);
        long layer2Size = (long) hiddenSize * feedForwardSize + hiddenSize;

        return qkvSize + projSize + normSize + layer1Size + layer2Size;
    }

    private int getIntProperty(Map<String, String> properties, String key) throws Exception
    {
        return toInt(getProperty(properties, key));
    }

    private float getFloatProperty(Map<String, String> properties, String key) throws Exception
    {
        return toFloat(getProperty(properties, key));
    }

    private String getProperty(Map<String, String> properties, String key) throws Exception
    {
        return getProperty(properties, key, false);
    }

    private String getProperty(Map<String, String> properties, String key, boolean isOptional) throws Exception
    {
        String value = properties.get(key);

        if (value != null)
        {
            value = value.trim();
        }

        if ( ! isOptional && (value == null || value.equals("")))
        {
            throw new Exception("Missing entry in the model.properties file: '" + key + "'.");
        }

        if (value != null && value.equals(""))
        {
            value = null;
        }

        return value;
    }

    private int toInt(String value) throws Exception
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to integer (" + value + ").");
        }
    }

    private float toFloat(String value) throws Exception
    {
        try
        {
            return Float.parseFloat(value);
        }
        catch (Exception e)
        {
            throw new Exception("The provided properties value can't be converted to float (" + value + ").");
        }
    }

    public String getConfigPath()
    {
        return arguments.getConfigRoot() + "/" + arguments.getName() + "/";
    }

    public String getModelPath()
    {
        return arguments.getModelRoot() + "/" + arguments.getName() + "/";
    }

    public String getName()
    {
        return name;
    }

    public String getTransformerType()
    {
        return transformerType;
    }

    public String getTokenizer()
    {
        return tokenizer;
    }

    public String getTokenizerConfig()
    {
        return tokenizerConfig;
    }

    public int getLengthLimit()
    {
        return arguments.getLengthLimit();
    }

    public int getTopK()
    {
        return arguments.getTopK();
    }

    public int getTokenCount()
    {
        return tokenCount;
    }

    public int getEndOfTextToken()
    {
        return endOfTextToken;
    }

    public int getMaxLength()
    {
        return maxLength;
    }

    public int getHiddenSize()
    {
        return hiddenSize;
    }

    public int getFeedForwardSize()
    {
        return feedForwardSize;
    }

    public int getDecoderCount()
    {
        return decoderCount;
    }

    public int getHeadCount()
    {
        return headCount;
    }

    public float getEpsilon()
    {
        return epsilon;
    }

    public String getParameterUrl()
    {
        return parameterUrl;
    }

    public List<String> getParameterFiles()
    {
        return parameterFiles;
    }

    public String getTransformerParameterFormat()
    {
        return transformerParameterFormat;
    }

    public String getDecoderParameterFormat()
    {
        return decoderParameterFormat;
    }

    public Map<String, String> getTransformerParameterOverrides()
    {
        return transformerParameterOverrides;
    }

    public Map<String, String> getDecoderParameterOverrides()
    {
        return decoderParameterOverrides;
    }

    public int getHeadSize()
    {
        return hiddenSize / headCount;
    }

    public ParameterReader getReader()
    {
        return reader;
    }

    public void setReader(ParameterReader reader)
    {
        this.reader = reader;
    }
}
