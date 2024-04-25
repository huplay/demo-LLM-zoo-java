package huplay.demo.config;

import java.util.List;

public class ParameterDescriptor
{
    private final String fileName;
    private final long dataOffset;
    private final String format;
    private final DataType dataType;
    private final List<Integer> shape;
    private final long startOffset;
    private final long endOffset;

    public ParameterDescriptor(String fileName, long dataOffset, String format, DataType dataType, List<Integer> shape,
                               long startOffset, long endOffset)
    {
        this.fileName = fileName;
        this.dataOffset = dataOffset;
        this.format = format;
        this.dataType = dataType;
        this.shape = shape;
        this.startOffset = startOffset;
        this.endOffset = endOffset;
    }

    public String getFileName()
    {
        return fileName;
    }

    public long getDataOffset()
    {
        return dataOffset;
    }

    public String getFormat()
    {
        return format;
    }

    public DataType getDataType()
    {
        return dataType;
    }

    public List<Integer> getShape()
    {
        return shape;
    }

    public long getStartOffset()
    {
        return startOffset;
    }

    public long getEndOffset()
    {
        return endOffset;
    }

    @Override
    public String toString()
    {
        return "ParameterDescriptor{" +
                "fileName='" + fileName + '\'' +
                ", format='" + format + '\'' +
                ", dataType='" + dataType + '\'' +
                ", shape=" + shape +
                ", startOffset=" + startOffset +
                ", endOffset=" + endOffset +
                '}';
    }
}
