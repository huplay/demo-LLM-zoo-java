package huplay.demo.transformer;

import huplay.demo.config.Config;
import huplay.demo.config.ParameterReader;
import huplay.demo.config.ParameterType;

import java.util.HashMap;
import java.util.Map;

public abstract class ParameterStore
{
    private final boolean isOnlyCalculation;
    private long parameterSize;

    protected final ParameterReader reader;
    protected final String parameterFormat;
    protected final Map<String, String> parameterOverrides;

    protected final Map<ParameterType, float[]> vectorParams = new HashMap<>();
    protected final Map<ParameterType, float[][]> matrixParams = new HashMap<>();

    public ParameterStore(Config config, String decoderParameterFormat, Map<String, String> parameterOverrides)
    {
        this.isOnlyCalculation = config.isCalculationOnly();
        this.reader = config.getReader();
        this.parameterFormat = decoderParameterFormat;
        this.parameterOverrides = parameterOverrides;
    }

    protected abstract String formatName(String file);

    protected void loadVector(ParameterType parameterType, String file, int size)
    {
        parameterSize += size;
        if (!isOnlyCalculation)
        {
            vectorParams.put(parameterType, reader.readVector(formatName(file), size));
        }
    }

    protected void loadVectorOptional(ParameterType parameterType, String file, int size)
    {
        parameterSize += size;
        if (!isOnlyCalculation)
        {
            vectorParams.put(parameterType, reader.readVectorOptional(formatName(file), size));
        }
    }

    protected void loadMatrix(ParameterType parameterType, String file, int rows, int cols)
    {
        parameterSize += (long) rows * cols;
        if (!isOnlyCalculation)
        {
            matrixParams.put(parameterType, reader.readMatrix(formatName(file), rows, cols));
        }
    }

    protected void loadMatrixOptional(ParameterType parameterType, String file, int rows, int cols)
    {
        parameterSize += (long) rows * cols;
        if (!isOnlyCalculation)
        {
            matrixParams.put(parameterType, reader.readMatrixOptional(formatName(file), rows, cols));
        }
    }

    public float[] vector(ParameterType parameterType)
    {
        return vectorParams.get(parameterType);
    }

    public float[][] matrix(ParameterType parameterType)
    {
        return matrixParams.get(parameterType);
    }

    public long getParameterSize()
    {
        return parameterSize;
    }
}
