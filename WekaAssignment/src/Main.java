
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.lazy.KStar;
import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.*;

import java.util.Random;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {
	// write your code here

        try
        {
            System.out.println("1. Thyroid Database");
            System.out.println("2. Glass Database ");
            System.out.print("\nSelect a Database : ");
            int choice = 1;
            Scanner in = new Scanner(System.in);
            choice = in.nextInt();

            String InputFileName = "";

            if(choice == 1)
                InputFileName = "new-thyroid.arff";
            else
                InputFileName = "glass.arff";

            DataSource source = new DataSource(InputFileName);
            Instances data = source.getDataSet();

            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);

            Evaluation eval = new Evaluation(data);

            System.out.println("1. Random Tree with Bagging");
            System.out.println("2. Random Forest ");
            System.out.println("3. (ANN)Multilayer Perceptron ");
            System.out.println("4. (SVM)SMOreg ");
            System.out.println("5. K Star ");
            System.out.print("\nSelect an Algorithm : ");

            choice = 0;
            in = new Scanner(System.in);
            choice = in.nextInt();

            System.out.println("For File: " + InputFileName);
            switch(choice) {

                case 1:

                    Bagging bag = new Bagging();
                    RandomTree rT = new RandomTree();

                    bag.setClassifier(rT);
                    bag.buildClassifier(data);

                    eval.crossValidateModel(bag, data, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\n----------Results of Random Tree with Bagging-----------\n", true));
                    break;

                case 2:
                    RandomForest rB = new RandomForest();
                    rB.buildClassifier(data);

                    eval.crossValidateModel(rB, data, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\n----------Results of Random Forest-----------\n", true));
                    break;

                case 3:
                    MultilayerPerceptron mlp = new MultilayerPerceptron();
                    mlp.buildClassifier(data);

                    eval.crossValidateModel(mlp, data, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\n----------Results of Multilayer Perceptron-----------\n", true));
                    break;

                case 4:
                    SMOreg sm = new SMOreg();
                    sm.buildClassifier(data);

                    eval.crossValidateModel(sm, data, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\n----------Results of Support Vector Machine-----------\n", true));
                    break;

                case 5:
                    KStar kS = new KStar();
                    kS.buildClassifier(data);

                    eval.crossValidateModel(kS, data, 10, new Random(1));
                    System.out.println(eval.toSummaryString("\n----------Results of K Star-----------\n", true));
                    break;

            }
        }
        catch (java.lang.Exception e)
        {
            e.printStackTrace();
        }
    }
}
