import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.apache.commons.io.FileUtils;

import weka.classifiers.functions.Logistic;
import weka.core.Instances;

public class DifficultyClassfierLR {
    public static void main(String args[]) throws Exception{
        //load train data
        Instances trainData = new Instances(new BufferedReader(new FileReader("Data/trainyt.arff")));
        trainData.setClassIndex(trainData.numAttributes() - 1);
     // Make the last attribute be the class 

        
        //load test data
        Instances testData = new Instances(new BufferedReader(new FileReader("Data/testnyt.arff")));
        testData.setClassIndex(testData.numAttributes() - 1);

        //build model
        Logistic model = new Logistic();
        model.buildClassifier(trainData); //the last instance with missing class is not used
        System.out.println(model);
        
        Instances labeled = new Instances(testData);
        
        String strg="";
        int c=0;
        for(int ii=0;ii<testData.numInstances();ii++)
        {
        double clsLabel = model.classifyInstance(testData.instance(ii));
        labeled.instance(ii).setClassValue(clsLabel);
      //  System.out.println(labeled);
        
        double[] distribution = model.distributionForInstance(testData.instance(ii));
        
        
        //in for khodash vojud dasht
      /*  for(int i = 0; i < distribution.length; i++) {
           System.out.println("distributionnnn "+distribution[i]);
        }*/

  
        //in for ra az inja peida kardam: http://www.programcreek.com/java-api-examples/index.php?api=weka.core.Instance 
        for (int i=0; i < testData.classAttribute().numValues(); ++i) {
        	 System.out.println(testData.classAttribute().value(i)+distribution[i]);
        	 c++;
        	 strg+=testData.classAttribute().value(i)+" "+distribution[i]+"\n";
          }
        FileUtils.writeStringToFile(new File("Data/res1.txt"), strg);
        System.out.println(c);
        }        
      }
}
