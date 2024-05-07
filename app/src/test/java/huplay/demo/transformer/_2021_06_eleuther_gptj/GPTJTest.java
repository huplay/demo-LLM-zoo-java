package huplay.demo.transformer._2021_06_eleuther_gptj;

import huplay.demo.config.Config;
import huplay.demo.transformer.BaseTransformerTest;
import org.junit.Ignore;
import org.junit.Test;

@Ignore
public class GPTJTest extends BaseTransformerTest
{
    @Test
    public void testTransformer()
    {
        Config config = getTestConfig("transformer/_2021_06_eleuther_gptj");

        GPTJ transformer = new GPTJ(config);

        // First run (no previously stored tokens)
        float[] result = transformer.execute(0, 0, true);

        float[] expected = new float[] {
                -1.043406f, 0.8742118f, 0.6131638f, -0.8746113f, -3.4446273f, 1.2662675f,
                1.7896787f, -6.4008756f, 3.8707275f, 1.5952247f, 0.17326549f, -0.10299438f,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

        testResult(expected, result, 0e-8f);

        // Second run
        result = transformer.execute(1, 1, true);

        expected = new float[] {
                0.5547548f, 4.879557f, -1.5965443f, -0.41758344f, 1.7790486f, -2.1507523f,
                1.1002446f, -6.2413406f, 4.086982f, -0.5690211f, 0.1137474f, -0.9097537f};

        testResult(expected, result, 0e-8f);
    }
}
