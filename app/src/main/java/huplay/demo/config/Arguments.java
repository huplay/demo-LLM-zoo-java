package huplay.demo.config;

import java.io.*;
import java.util.*;

/**
 * Holder of the app's input parameters
 */
public class Arguments
{
    private static final String ARG_CALC = "-calc";
    private static final String ARG_MAX = "-max";
    private static final String ARG_TOP_K = "-topK";
    private static final String ARG_MEM = "-mem";

    // The root folder of the model configurations
    // The default is the modelConfig, but it can be overridden by the DEMO_LLM_ZOO_CONFIG_ROOT environment variable
    private final String configRoot;

    // The root folder of the model parameters
    // The default is the modelConfig, but it can be overridden by the DEMO_LLM_ZOO_MODEL_ROOT environment variable
    private final String modelRoot;

    // The relative path of the selected model
    private String relativePath;

    private final int lengthLimit;
    private final int topK;
    private boolean isCalculationOnly;
    private final Integer requestedMemorySize;

    public Arguments(String configRoot, String modelRoot, String relativePath,
                     int lengthLimit, int topK, boolean isCalculationOnly, int requestedMemorySize)
    {
        this.configRoot = configRoot;
        this.modelRoot = modelRoot;
        this.relativePath = relativePath;
        this.lengthLimit = lengthLimit;
        this.topK = topK;
        this.isCalculationOnly = isCalculationOnly;
        this.requestedMemorySize = requestedMemorySize;
    }

    public static Arguments readArguments(String[] args)
    {
        File file = new File("modelConfig");
        String configRoot = System.getenv().getOrDefault("DEMO_LLM_ZOO_CONFIG_ROOT", file.getAbsolutePath());
        String modelRoot = System.getenv().getOrDefault("DEMO_LLM_ZOO_MODEL_ROOT", configRoot);

        configRoot = configRoot.replace("\\", "/");
        modelRoot = modelRoot.replace("\\", "/");

        // Default values
        String modelPath = null;
        int maxLength = 25;
        int topK = 40;
        int requestedMemorySize = 0;
        boolean isCalculationOnly = false;

        if (args != null)
        {
            // Iterate over the passed parameters and override the default values
            for (String arg : args)
            {
                if (arg.charAt(0) == '-')
                {
                    if (equals(arg, ARG_CALC)) isCalculationOnly = true;
                    else
                    {
                        String[] parts = arg.split("=");
                        if (parts.length == 2)
                        {
                            String key = parts[0];
                            String value = parts[1];

                            if (equals(key, ARG_MAX)) maxLength = readInt(value, maxLength);
                            else if (equals(key, ARG_TOP_K)) topK = readInt(value, topK);
                            else if (equals(key, ARG_MEM)) requestedMemorySize = readInt(value, 0);
                        }
                        else
                        {
                            System.out.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                        }
                    }
                }
                else if (modelPath != null)
                {
                    System.out.println("\nWARNING: Unrecognisable argument: " + arg + "\n");
                }
                else
                {
                    modelPath = removeDoubleQuotes(arg);
                }
            }
        }

        return new Arguments(configRoot, modelRoot, modelPath, maxLength, topK, isCalculationOnly, requestedMemorySize);
    }

    // Getters, setters
    public String getConfigRoot() {return configRoot;}
    public String getModelRoot() {return modelRoot;}
    public String getRelativePath() {return relativePath;}
    public int getLengthLimit() {return lengthLimit;}
    public int getTopK() {return topK;}
    public boolean isCalculationOnly() {return isCalculationOnly;}
    public Integer getRequestedMemorySize() {return requestedMemorySize;}

    // Setters
    public void setRelativePath(String relativePath) {this.relativePath = relativePath;}
    public void setCalculationOnly(boolean calculationOnly) {isCalculationOnly = calculationOnly;}

    public String getConfigPath()
    {
        return relativePath == null ? null : configRoot + "/" + relativePath;
    }

    public String getModelPath()
    {
        return relativePath == null ? null : modelRoot + "/" + relativePath;
    }

    private static boolean equals(String a, String b)
    {
        return a.toLowerCase(Locale.ROOT).equals(b.toLowerCase(Locale.ROOT));
    }

    private static int readInt(String value, int defaultValue)
    {
        try
        {
            return Integer.parseInt(value);
        }
        catch (Exception e)
        {
            System.out.println("\nWARNING: The provided value can't be converted to integer (" + value
                    + "). Default value will be used.\n");
        }

        return defaultValue;
    }

    private static String removeDoubleQuotes(String text)
    {
        if (text == null) return null;
        if (text.charAt(0) == '"') text = text.substring(1);
        if (text.charAt(text.length() - 1) == '"') text = text.substring(0, text.length() - 1);
        return text;
    }
}