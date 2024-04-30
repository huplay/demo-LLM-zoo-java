package huplay.demo;

import huplay.demo.config.Config;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

public class ConfigTest
{
    @Test
    public void testAddProperty()
    {
        checkAdded("key = value", "key", "value");
        checkAdded("key=value", "key", "value");
        checkAdded("  key=value", "key", "value");
        checkAdded("key= value", "key", "value");
        checkAdded("key= value", "key", "value");
        checkAdded("key= value = x", "key", "value = x");
        checkAdded("   key    =    value    ", "key", "value");
        checkAdded("   key    =    value    x", "key", "value    x");

        checkNotAdded(null);
        checkNotAdded("");
        checkNotAdded(" ");
        checkNotAdded("#comment");
        checkNotAdded("# comment");
        checkNotAdded("key=   ");
        checkNotAdded("key =");
        checkNotAdded("key = ");
        checkNotAdded("   key = ");

        System.out.println("The following 12 warnings are added by a unit test, it's ok");
        checkNotAdded("=");
        checkNotAdded("=    ");
        checkNotAdded("   =");
        checkNotAdded(" =  ");
        checkNotAdded("   # ");
        checkNotAdded("   # comment");
        checkNotAdded("value");
        checkNotAdded("  value");
        checkNotAdded("value   ");
        checkNotAdded("=value");
        checkNotAdded("   =value");
        checkNotAdded("   =  value  ");
    }

    private void checkAdded(String line, String key, String value)
    {
        Map<String, String> properties = new HashMap<>();

        Config.addProperty(line, properties);

        String storedValue = properties.get(key);
        assertNotNull(storedValue);
        assertEquals(storedValue, value);
    }

    private void checkNotAdded(String line)
    {
        Map<String, String> properties = new HashMap<>();

        Config.addProperty(line, properties);

        assertEquals(0, properties.size());
    }
}
