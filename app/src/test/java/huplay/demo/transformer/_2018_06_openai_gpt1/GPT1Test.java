package huplay.demo.transformer._2018_06_openai_gpt1;

import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformerTest;
import huplay.demo.util.Vector;
import org.junit.Test;

public class GPT1Test extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        Config config = getTestConfig("transformer/_2018_06_openai_gpt1");

        GPT1 transformer = new GPT1(config);

        // First run (no previously stored tokens)
        Vector result = transformer.execute(0, 0, true);

        float[] expected = new float[] {
                0.30690542f, -0.9076174f, 0.506058f, 0.22302012f, 0.15793128f, -0.7363872f,
                0.22478494f, 0.31726304f, -0.121716924f, -0.43617797f, -0.3724439f, 0.31970194f};

        assertVectorEquals(expected, result, 1e-6f);

        // Second run
        result = transformer.execute(1, 1, true);

        expected = new float[] {
                0.009709928f, -0.8714771f, 0.20586778f, 0.40443808f, 0.24443056f, -0.52534175f,
                -0.0033788234f, 0.6607293f, 0.25798416f, -0.5441957f, -0.51625514f, 0.14530993f};

        assertVectorEquals(expected, result, 1e-6f);
    }
}
