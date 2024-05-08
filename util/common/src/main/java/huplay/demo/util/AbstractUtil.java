package huplay.demo.util;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public abstract class AbstractUtil implements Utility
{
    @Override
    public float max(Vector vector)
    {
        float max = Float.NEGATIVE_INFINITY;

        for (int i = 0; i < vector.size(); i++)
        {
            float value = vector.get(i);
            if (value > max)
            {
                max = value;
            }
        }

        return max;
    }

    @Override
    public float max(List<IndexedValue> vector)
    {
        float max = Float.NEGATIVE_INFINITY;

        for (IndexedValue indexedValue : vector)
        {
            if (indexedValue.getValue() > max)
            {
                max = indexedValue.getValue();
            }
        }

        return max;
    }

    @Override
    public Vector normalize(Vector vector, float epsilon)
    {
        float average = average(vector);
        float averageDiff = averageDiff(vector, average, epsilon);

        Vector norm = new Vector(vector.getFloatType(), vector.size());

        for (int i = 0; i < vector.size(); i++)
        {
            norm.set(i, (vector.get(i) - average) / averageDiff);
        }

        return norm;
    }

    @Override
    public float averageDiff(Vector values, float average, float epsilon)
    {
        Vector squareDiff = new Vector(values.getFloatType(), values.size());

        for (int i = 0; i < values.size(); i++)
        {
            float diff = values.get(i) - average;
            squareDiff.set(i, diff * diff);
        }

        float averageSquareDiff = average(squareDiff);

        return (float) Math.sqrt(averageSquareDiff + epsilon);
    }


    /**
     * Sort values to reversed order and filter out the lowest values (retain the top [count] values)
     */
    public List<IndexedValue> reverseAndFilter(float[] values, int count)
    {
        TreeSet<IndexedValue> indexedValues = new TreeSet<>(new IndexedValue.ReverseComparator());
        for (int i = 0; i < values.length; i++)
        {
            indexedValues.add(new IndexedValue(values[i], i));
        }

        List<IndexedValue> filteredValues = new ArrayList<>(count);

        int i = 0;
        for (IndexedValue indexedValue : indexedValues)
        {
            filteredValues.add(indexedValue);
            i++;
            if (i == count) break;
        }

        return filteredValues;
    }
}
