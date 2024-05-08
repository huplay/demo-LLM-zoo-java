package huplay.demo.transformer;

import huplay.demo.BaseTest;
import huplay.demo.config.Arguments;
import huplay.demo.config.Config;
import huplay.demo.config.ModelConfig;
import huplay.demo.config.ParameterReader;

import java.io.File;

public class BaseTransformerTest extends BaseTest
{
    protected Config getTestConfig(String relativePath)
    {
        File resourcesDirectory = new File("src/test/resources");
        String root = resourcesDirectory.getAbsolutePath();

        Arguments arguments = new Arguments(root, root, relativePath, 25, 40,
                false, 0);

        ModelConfig modelConfig = ModelConfig.read(arguments);

        ParameterReader reader = new ParameterReader(arguments.getModelPath());
        return Config.readConfig(arguments, modelConfig, reader);
    }
}
