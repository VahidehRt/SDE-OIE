import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

import org.apache.commons.io.FileUtils;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class DifficultyClassfierNB {
    public static void main(String[] args) throws Exception {

        //
        // Load train data
        // 
        String readTrain = "DifficultyDataSet/NB/trainnytweb2refinrmv.arff";
        BufferedReader readerTrain = new BufferedReader(new FileReader(readTrain));
        Instances train = new Instances(readerTrain);
        readerTrain.close();
        train.setClassIndex(train.numAttributes() - 1);         

        //
        // Load test data
        // 
        String readTest = "DifficultyDataSet/NB/testnytweb2refinrmv.arff";
        BufferedReader readerTest = new BufferedReader(new FileReader(readTest));
        Instances test = new Instances(readerTest);
        readerTest.close();
        test.setClassIndex(test.numAttributes() - 1);  

        // Create a naïve bayes classifier
        Classifier cModel = (Classifier)new NaiveBayes();
        cModel.buildClassifier(train);
        System.out.println(cModel);


        // Predict distribution of instance
        double[] fDistribution = cModel.distributionForInstance(test.instance(2));
       // System.out.println("Prediction class 1: " + fDistribution[0]);
        //System.out.println("Prediction class 2: " + fDistribution[1]);
        
        
        
        Instances labeled = new Instances(test);
        
        String strg="";
        int c=0;
        for(int ii=0;ii<test.numInstances();ii++)
        {
        double clsLabel = cModel.classifyInstance(test.instance(ii));
        labeled.instance(ii).setClassValue(clsLabel);
      //  System.out.println(labeled);
        
        double[] distribution = cModel.distributionForInstance(test.instance(ii));
  
     
        for (int i=0; i < test.classAttribute().numValues(); ++i) {
        	 System.out.println(test.classAttribute().value(i)+distribution[i]);
        	 c++;
        	 strg+=test.classAttribute().value(i)+" "+distribution[i]+"\n";
          }
        FileUtils.writeStringToFile(new File("DifficultyDataSet/NB/NaByz.txt"), strg);
        System.out.println(c);
        }          
              
        
    }
}