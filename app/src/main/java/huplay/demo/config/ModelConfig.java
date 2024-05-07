package huplay.demo.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import huplay.demo.IdentifiedException;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Holder of the configuration stored in the model.json file
 * This is the file to describe where is the model, how to download, which file are needed, what is the config file name
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig
{
    private Arguments arguments;

    private String name;
    private String transformerType;
    private String tokenizerType;
    private String repo;
    private String branch;
    private List<String> files;
    private Map<String, String> fileNameOverrides;
    private String parameterNaming;
    private String decoderParameterNaming;
    private Map<String, String> parameterNameOverrides;
    private Integer memorySize;

    public static ModelConfig read(Arguments arguments)
    {
        String modelConfigJson = arguments.getConfigPath() + "/model.json";

        File modelConfigFile = new File(modelConfigJson);
        if (!modelConfigFile.exists())
        {
            throw new IdentifiedException("Model config file is missing (" + modelConfigJson + ")");
        }

        try
        {
            ModelConfig modelConfig = new ObjectMapper().readValue(modelConfigFile, ModelConfig.class);
            modelConfig.init(arguments);

            return modelConfig;
        }
        catch (IOException e)
        {
            throw new IdentifiedException("Cannot read model.json (" + modelConfigJson + ")");
        }
    }

    private void init(Arguments arguments)
    {
        this.arguments = arguments;

        if (fileNameOverrides == null) fileNameOverrides = new HashMap<>();

        if (parameterNaming == null) parameterNaming = "{name}";

        if (decoderParameterNaming == null) decoderParameterNaming = "{decoderId}.{name}";

        if (parameterNameOverrides == null) parameterNameOverrides = new HashMap<>();
    }

    // Getters
    public String getName() {return name;}
    public String getTransformerType() {return transformerType;}
    public String getTokenizerType() {return tokenizerType;}
    public String getRepo() {return repo;}
    public String getBranch() {return branch;}
    public List<String> getFiles() {return files;}
    public Map<String, String> getFileNameOverrides() {return fileNameOverrides;}
    public String getParameterNaming() {return parameterNaming;}
    public String getDecoderParameterNaming() {return decoderParameterNaming;}
    public Map<String, String> getParameterNameOverrides() {return parameterNameOverrides;}
    public Integer getMemorySize() {return memorySize;}

    public String resolveFileName(String name)
    {
        String overriddenName = fileNameOverrides.get(name);
        return overriddenName == null ? name : overriddenName;
    }

    public File findFile(String name)
    {
        String resolvedName = resolveFileName(name);
        File file = new File(arguments.getModelPath() + "/" + resolvedName);
        if (file.exists() && file.isFile())
        {
            return file;
        }
        else
        {
            return new File(arguments.getConfigPath() + "/" + resolvedName);
        }
    }
}
