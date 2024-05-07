package huplay.demo.transformer;

import huplay.demo.config.Arguments;
import huplay.demo.config.Config;
import huplay.demo.config.ModelConfig;
import huplay.demo.config.ParameterReader;

import java.io.File;

import static org.junit.Assert.*;

public class BaseTransformerTest
{
    protected Config getTestConfig(String relativePath)
    {
        File resourcesDirectory = new File("src/test/resources");
        String root = resourcesDirectory.getAbsolutePath();

        Arguments arguments = new Arguments(root, root, relativePath, 25, 40,
                false, 0);

        ModelConfig modelConfig = ModelConfig.read(arguments);

        /*ModelConfig modelConfig = new ModelConfig();
        modelConfig.init(arguments);
        modelConfig.setNameFormat(nameFormat);
        modelConfig.setDecoderParameterNaming(decoderNameFormat);*/

        ParameterReader reader = new ParameterReader(arguments.getModelPath());
        return Config.readConfig(arguments, modelConfig, reader);
    }

    protected void testResult(float[] expected, float[] actual, float delta)
    {
        assertNotNull(expected);
        assertNotNull(actual);

        assertEquals(expected.length, actual.length);

        for (int i = 0; i < expected.length; i++)
        {
            assertEquals(expected[i], actual[i], delta);
        }
    }
}
