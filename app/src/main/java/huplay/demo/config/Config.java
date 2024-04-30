package huplay.demo.config;

import java.io.*;
import java.util.*;

import static huplay.demo.AppMain.OUT;

/**
 * Holder of the configuration stored in the model.properties file
 */
public class Config
{
    private final Map<String, String> properties;
    private boolean isCalculationOnly;
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

    private final String parameterRepo;
    private final String parameterRepoBranch;
    private final List<String> parameterFiles;
    private final String transformerParameterFormat;
    private final String decoderParameterFormat;
    private final Map<String, String> transformerParameterOverrides;
    private final Map<String, String> decoderParameterOverrides;

    private final int memorySizeTotal;
    private final int memorySizeAdditional;

    private ParameterReader reader;

    public Config(Arguments arguments) throws Exception
    {
        this.arguments = arguments;
        this.isCalculationOnly = arguments.isCalculationOnly();

        // Read all properties from the model.properties file
        this.properties = readProperties(getConfigPath() + "model.properties");
        this.name = getProperty("name");
        this.transformerType = getProperty("transformer.type");
        this.tokenizer = getProperty("tokenizer");
        this.tokenizerConfig = getProperty("tokenizer.config");
        this.tokenCount = getIntProperty("token.count");
        this.endOfTextToken = getIntProperty("end.of.text.token");
        this.maxLength = getIntProperty("max.length");

        this.hiddenSize = getIntProperty("hidden.size");
        this.feedForwardSize = getIntProperty("feedforward.size");
        this.decoderCount = getIntProperty("decoder.count");
        this.headCount = getIntProperty("attention.head.count");
        this.epsilon = getFloatProperty("epsilon");

        this.parameterRepo = getProperty("parameter.repo", true);
        this.parameterRepoBranch = getProperty("parameter.repo.branch", true);
        this.parameterFiles = readParameterFiles();
        this.transformerParameterFormat = properties.get("transformer.parameter.format");
        this.decoderParameterFormat = properties.get("decoder.parameter.format");
        this.transformerParameterOverrides = readParameterOverrides("transformer.parameter.overrides");
        this.decoderParameterOverrides = readParameterOverrides("decoder.parameter.overrides");

        this.memorySizeTotal = getIntPropertyOptional("memory.size.total", 0);
        this.memorySizeAdditional = getIntPropertyOptional("memory.size.additional", 0);
    }

    public int getIntProperty(String key)
    {
        try
        {
            return toInt(getProperty(key));
        }
        catch (Exception e)
        {
            throw new RuntimeException("Cannot read integer property: " + key + " exception: " + e.getMessage());
        }
    }

    public int getIntPropertyOptional(String key, int defaultValue)
    {
        try
        {
            String property = getProperty(key, true);
            return property == null ? defaultValue : toInt(property);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Cannot read integer property: " + key + " exception: " + e.getMessage());
        }
    }

    public float getFloatProperty(String key)
    {
        try
        {
            return toFloat(getProperty(key));
        }
        catch (Exception e)
        {
            throw new RuntimeException("Cannot read float property: " + key + " exception: " + e.getMessage());
        }
    }

    public float getFloatPropertyOptional(String key, float defaultValue) throws Exception
    {
        try
        {
            String property = getProperty(key, true);
            return property == null ? defaultValue : toFloat(property);
        }
        catch (Exception e)
        {
            throw new RuntimeException("Cannot read float property: " + key + " exception: " + e.getMessage());
        }
    }

    public String getProperty(String key)
    {
        return getProperty(key, false);
    }

    public String getPropertyOptional(String key, String defaultValue)
    {
        String property = getProperty(key, true);
        return property == null ? defaultValue : property;
    }

    private String getProperty(String key, boolean isOptional)
    {
        String value = properties.get(key);

        if (value != null)
        {
            value = value.trim();
        }

        if ( ! isOptional && (value == null || value.equals("")))
        {
            throw new RuntimeException("Missing entry in the model.properties file: '" + key + "'.");
        }

        if (value != null && value.equals(""))
        {
            value = null;
        }

        return value;
    }

    public boolean isCalculationOnly()
    {
        return isCalculationOnly;
    }

    public void setCalculationOnly(boolean calculationOnly)
    {
        isCalculationOnly = calculationOnly;
    }

    private List<String> readParameterFiles()
    {
        return splitProperty(getProperty("parameter.files"));
    }

    private Map<String, String> readParameterOverrides(String name)
    {
        String property = getProperty(name, true);

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

    private static Map<String, String> readProperties(String fileName) throws Exception
    {
        Map<String, String> properties = new HashMap<>();

        try (Scanner scanner = new Scanner(new File(fileName)))
        {
            while (scanner.hasNextLine())
            {
                addProperty(scanner.nextLine(), properties);
            }
        }
        catch (IOException e)
        {
            throw new Exception("Cannot read model.properties file: " + fileName);
        }

        return properties;
    }

    public static void addProperty(String line, Map<String, String> properties)
    {
        if (line != null && !line.trim().equals("") && !line.startsWith("#"))
        {
            String trimmedLine = line.trim();
            int index = trimmedLine.indexOf('=');
            if (index > 0)
            {
                String key = trimmedLine.substring(0, index).trim();
                String value = trimmedLine.substring(index + 1).trim();

                if (!key.equals("") && !value.equals(""))
                {
                    properties.put(key, value);
                }

                if (key.equals("") && !value.equals(""))
                {
                    OUT.println("\nWARNING: Incorrectly formed line found in the properties file: \"" + line + "\"");
                }
            }
            else
            {
                OUT.println("\nWARNING: Incorrectly formed line found in the properties file: \"" + line + "\"");
            }
        }
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

    public int getMemorySize()
    {
        return arguments.getRequestedMemorySize();
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

    public String getParameterRepo()
    {
        return parameterRepo;
    }

    public String getParameterRepoBranch()
    {
        return parameterRepoBranch;
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

    public int getMemorySizeTotal()
    {
        return memorySizeTotal;
    }

    public int getMemorySizeAdditional()
    {
        return memorySizeAdditional;
    }
}
