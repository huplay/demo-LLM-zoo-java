package huplay.demo.transformer._2019_02_openai_gpt2;

import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformerTest;

import huplay.demo.util.Vector;
import org.junit.Test;

public class GPT2Test extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        Config config = getTestConfig("transformer/_2019_02_openai_gpt2");

        GPT2 transformer = new GPT2(config);

        // First run (no previously stored tokens)
        Vector result = transformer.execute(0, 0, true);

        float[] expected = new float[] {
                0.27687562f, -0.28724107f, 1.0454319f, 0.7014351f, 1.1113691f, -1.2056924f,
                -15.5616865f, -1.1972752f, 2.7882829f, 0.0032621801f, -1.763285f, 0.53614736f};

        assertVectorEquals(expected, result, 1e-6f);

        // Second run
        result = transformer.execute(1, 1, true);

        expected = new float[] {
                1.4605376f, 1.7202338f, -0.79042673f, -1.6437954f, -0.06698311f, -2.2982268f,
                -7.197687f, 0.44757664f, 1.4767196f, 0.0032621801f, -1.1588063f, 0.76185316f};

        assertVectorEquals(expected, result, 1e-6f);
    }
}
