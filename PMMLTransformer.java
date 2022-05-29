import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

import javax.xml.parsers.ParserConfigurationException;

import jakarta.xml.bind.JAXBException;

import org.dmg.pmml.PMML;
import org.jpmml.model.SerializationUtil;
import org.jpmml.model.visitors.LocatorNullifier;
import org.xml.sax.SAXException;

public class PMMLTransformer {
    private final static String BASE_PATH = "C:\\Users\\rauls\\Downloads";

    public static void main(String[] args) throws Exception {
        File pmmlFile = new File(BASE_PATH + "\\model.pmml");
        File serFile = new File(BASE_PATH + "\\model.pmml.ser");

        // Parse a pmml object from a file
        PMML pmml;
        try (InputStream is = new FileInputStream(pmmlFile)) {
        	System.out.println(pmmlFile.exists());
            pmml = load(is);
        }

        optimize(pmml);

        // Write an ser file from the pmml object
        try (OutputStream os = new FileOutputStream(serFile)) {
            SerializationUtil.serializePMML(pmml, os);
        }
    }
    
    public static PMML load(InputStream is) throws SAXException, JAXBException {
    	  try {
			return org.jpmml.model.PMMLUtil.unmarshal(is);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (JAXBException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
    	}
    
    public static void optimize(PMML pmml){
    	  LocatorNullifier nullifier = new LocatorNullifier();
    	  nullifier.applyTo(pmml);
    	}

}



