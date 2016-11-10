package javabbob;

import java.text.DecimalFormat;

import java.util.Random;
import java.util.Calendar;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;

import java.lang.Math.*;

/** Wrapper class running an entire BBOB experiment.
 * It illustrates the benchmarking of MY_OPTIMIZER on the noise-free testbed
 * or the noisy testbed (change the ifun loop in this case as given below).
 * This class reimplements the content of exampleexperiment.c from the original
 * C version of the BBOB code.
 */
public class ExampleExperiment {

    static DecimalFormat doubleFormatter = new DecimalFormat("#0.00");

    // This is used to determine which values can be taken by the coordinates of input vectors 
    static double RANGE_INTERVAL = 0.25;

    // The 24 Coco functions are defined on [5;5]
    static int MIN_RANGE_VALUE = -5;
    static int MAX_RANGE_VALUE = 5;

    //Report abbreviation : NIV
    //This is the size of the sample data set of input / output
    static int NUMBER_OF_INPUT_VECTORS = 20;  

    // Report abbreviation : NHL
    static int NUMBER_OF_HIDDEN_LAYERS = 10; 

    //Report abbreviation : NNPL
    static int NUMBER_OF_NEURONS_PER_HIDDEN_LAYER = 8;
    
    //Report abbreviation : NC
    // Size of the initial MLP Population
    static int NUMBER_OF_CHROMOSOMES = 30;

    /** Example optimiser.
     * In the following, the pure random search optimization method is
     * implemented as an example.
     * Please include/insert any code as suitable.<p>
     * This optimiser takes as argument an instance of JNIfgeneric
     * which have all the information on the problem to solve.
     * Only the methods getFtarget() and evaluate(x) of the class JNIfgeneric
     * are used.<p>
     * This method also takes as argument an instance of Random since one
     * might want to set the seed of the random search.<p>
     * The optimiser generates random vectors evaluated on fgeneric until
     * the number of function evaluations is greater than maxfunevals or
     * a function value smaller than the target given by fgeneric.getFtarget()
     * is attained.
     * The parameter maxfunevals to avoid problem when comparing it to
     * 1000000000*dim where dim is the dimension of the problem.
     * @param fgeneric an instance JNIfgeneric object
     * @param dim an integer giving the dimension of the problem
     * @param maxfunevals the maximum number of function evaluations
     * @param rand an instance of Random
     */
    public static void MY_OPTIMIZER(JNIfgeneric fgeneric, int dim, double maxfunevals) {


        /* Obtain the target function value, which only use is termination */
        double ftarget = fgeneric.getFtarget();
        double f;


            
                /*  ------------------------------------------------------------------------------------------------------------------------- */
                /*  --------------------------------------------- Step 0 -------------------------------------------------------------------- */
                /*  ------------------------------------------------------------------------------------------------------------------------- */
                // step 0 : input output table creation to approximate the function

                // contains input for function ifun number 1. Range : [-5;5], Interval : RANGE_INTERVAL
                double[] knownInputValuesTable = generateKnownInput();

                // This is used to approximate the function
                // We add +1 to have the output of the function in the same table as input (in the last column / index)
                // contains both input and output of function ifun number 1. Range : [-5;5], Interval : RANGE_INTERVAL
                double[][] sampleInputDataSet = generateSampleInputData(dim+1,knownInputValuesTable);

                /*  ------------------------------------------------------------------------------------------------------------------------- */
                /*  --------------------------------------------- Step 1 -------------------------------------------------------------------- */
                /*  ------------------------------------------------------------------------------------------------------------------------- */
                // step 1 : creation of initial chromosome population

                // Creation of the chromosome population
                ArrayList<double[]> initialChromosomePopulation = generatePopulation(dim);
                //displayChromosomes(initialChromosomePopulation);

                /*  ------------------------------------------------------------------------------------------------------------------------- */
                /*  --------------------------------------------- Step 2 -------------------------------------------------------------------- */
                /*  ------------------------------------------------------------------------------------------------------------------------- */
                // step 2 : calculating output of each chromosome of the initial population AND fitness of population

                // This contains the initial population outputs for each input
                // When creating our GA outside Coco we needed to store the output of the function
                ArrayList<double[]> mLPPopulationOutput = getMLPPopulationOutput(sampleInputDataSet,initialChromosomePopulation);
                //displayChromosomePopulationOutput(mLPPopulationOutput);

                // Calculation of initial population fitness
                double currentGenerationFitness = getPopulationFitness(sampleInputDataSet,mLPPopulationOutput);
                //System.out.println("Population fitness : " + currentGenerationFitness);

                /*  ------------------------------------------------------------------------------------------------------------------------- */
                /*  --------------------------------------------- Step 3 -------------------------------------------------------------------- */
                /*  ------------------------------------------------------------------------------------------------------------------------- */
                
                // Variables for current generation

                // Parent generation
                ArrayList<double[]> generationToKeep = initialChromosomePopulation;

                // Parent generation output
                ArrayList<double[]> generationToKeepOutput = mLPPopulationOutput;

                // Parent generation fitness
                double generationToKeepFitness = currentGenerationFitness;


                // Modified parent generation (on which we apply crossover and mutation) (not T+1 generation)
                ArrayList<double[]> modifiedCurrentGeneration = new ArrayList<double[]>();

                // Modified parent generation output
                ArrayList<double[]> modifiedCurrentGenerationOutput = new ArrayList<double[]>();

                // Modified parent generation fitness
                double modifiedCurrentGenerationFitness = 0;


                // Main loop
                do{

                    // Selecting parents, doing crossover and mutation
                    modifiedCurrentGeneration = generateNextGenerationPopulation(generationToKeep,generationToKeepOutput);
                    //displayChromosomes(modifiedCurrentGeneration);
               

                    // Calculating fitness of the modified generation
                    modifiedCurrentGenerationOutput = getMLPPopulationOutput(sampleInputDataSet,modifiedCurrentGeneration);
                    modifiedCurrentGenerationFitness = getPopulationFitness(sampleInputDataSet,modifiedCurrentGenerationOutput);
                   

                    //System.out.println("Raw generation fitness :" + generationToKeepFitness);
                    //System.out.println("Modified generation fitness :" + modifiedCurrentGenerationFitness);

                    // Here our fitness means having low error : we take the modified generation if it has smaller fitness
                    if (generationToKeepFitness > modifiedCurrentGenerationFitness) {

                        //System.out.println("Taking modified generation");
                        generationToKeep = modifiedCurrentGeneration;
               
                    }


                    // Removing an individual from the best chosen population (chosen in the previous step)
                    // This is T+1 generation (child generation)
                    //generationToKeep = removeWorstIndividual(generationToKeep,modifiedCurrentGenerationOutput); // this removes the worst individual according to custom fitness
                    generationToKeep = removeRandomIndividual(generationToKeep);

                    // Updating T+1 generation output and fitness
                    generationToKeepOutput = getMLPPopulationOutput(sampleInputDataSet,generationToKeep);
                    generationToKeepFitness = getPopulationFitness(sampleInputDataSet,generationToKeepOutput);


                }while (generationToKeep.size() > 1);

                //System.out.println("Best chromosome is the following : ");
                //displayChromosomes(generationToKeep);
                //System.out.println("It has a fitness of : " + getPopulationFitness(sampleInputDataSet,generationToKeepOutput));


                // COMPARING REAL OUTPUT AND THIS CHROMOSOME OUTPUT
                //System.out.println("MLP Output :");
                //displayChromosomePopulationOutput(generationToKeepOutput);
                //System.out.println("Final population fitness : " + getChromosomeFitness(bestChromosome));

            /* evaluate the best chromosome on the objective function */
            f = fgeneric.evaluate(generationToKeep.get(0));

          
        
        
    }

    /** Main method for running the whole BBOB experiment.
     *  Executing this method runs the experiment.
     *  The first command-line input argument is interpreted: if given, it
     *  denotes the data directory to write in (in which case it overrides
     *  the one assigned in the preamble of the method).
     */
    public static void main(String[] args) {

        /* run variables for the function/dimension/instances loops */
        final int dim[] = {2, 3, 5, 10, 20, 40};
        final int instances[] = {1, 2, 3, 4, 5, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50};
        int idx_dim, ifun, idx_instances, independent_restarts;
        double maxfunevals;
        String outputPath;

        JNIfgeneric fgeneric = new JNIfgeneric();
        /* The line above loads the library cjavabbob at the core of
         * JNIfgeneric. It will throw runtime errors if the library is not
         * found.
         */

        /**************************************************
         *          BBOB Mandatory initialization         *
         *************************************************/
        JNIfgeneric.Params params = new JNIfgeneric.Params();
        /* Modify the following parameters, choosing a different setting
         * for each new experiment */
        params.algName = "PUT ALGORITHM NAME";
        params.comments = "PUT MORE DETAILED INFORMATION, PARAMETER SETTINGS ETC";
        outputPath = "PUT_MY_BBOB_DATA_PATH";

        if (args.length > 0) {
            outputPath = args[0]; // Warning: might override the assignment above.
        }

        /* Creates the folders for storing the experimental data. */
        if ( JNIfgeneric.makeBBOBdirs(outputPath, false) ) {
            System.out.println("BBOB data directories at " + outputPath
                    + " created.");
        } else {
            System.out.println("Error! BBOB data directories at " + outputPath
                    + " was NOT created, stopping.");
            return;
        };


        /* External initialization of MY_OPTIMIZER */
        long seed = System.currentTimeMillis();
        Random rand = new Random(seed);
        System.out.println("MY_OPTIMIZER seed: "+ seed);

        /* record starting time (also useful as random number generation seed) */
        long t0 = System.currentTimeMillis();

        /* To make the noise deterministic, uncomment the following block. */
        /* int noiseseed = 30; // or (int)t0
         * fgeneric.setNoiseSeed(noiseseed);
         * System.out.println("seed for the noise set to: "+noiseseed); */

        /* now the main loop */
        for (idx_dim = 0; idx_dim < 6; idx_dim++) {
            /* Function indices are from 1 to 24 (noiseless) or from 101 to 130 (noisy) */
            /* for (ifun = 101; ifun <= 130; ifun++) { // Noisy testbed */
            for (ifun = 1; ifun <= 24; ifun++) { //Noiseless testbed
                System.out.println("Starting loop for function number " + ifun);
                for (idx_instances = 0; idx_instances < 15; idx_instances++) {

                    /* Initialize the objective function in fgeneric. */
                    fgeneric.initBBOB(ifun, instances[idx_instances],
                                      dim[idx_dim], outputPath, params);
                    /* Call to the optimizer with fgeneric as input */
                    maxfunevals = 5. * dim[idx_dim]; /* PUT APPROPRIATE MAX. FEVALS */
                                                     /* 5. * dim is fine to just check everything */
                    independent_restarts = -1;
                    while (fgeneric.getEvaluations() < maxfunevals) {
                        independent_restarts ++;
                        MY_OPTIMIZER(fgeneric, dim[idx_dim],
                                     maxfunevals - fgeneric.getEvaluations());
                        if (fgeneric.getBest() < fgeneric.getFtarget())
                            break;
                    }

                    System.out.printf("  f%d in %d-D, instance %d: FEs=%.0f with %d restarts,", ifun, dim[idx_dim],
                                      instances[idx_instances], fgeneric.getEvaluations(), independent_restarts);
                    System.out.printf(" fbest-ftarget=%.4e, elapsed time [h]: %.2f\n", fgeneric.getBest() - fgeneric.getFtarget(),
                                      (double) (System.currentTimeMillis() - t0) / 3600000.);

                    /* call the BBOB closing function to wrap things up neatly */
                    fgeneric.exitBBOB();
                }

                System.out.println("\ndate and time: " + (new SimpleDateFormat("dd-MM-yyyy HH:mm:ss")).format(
                        (Calendar.getInstance()).getTime()));
                System.out.println("function " + ifun + " done");

            }
            System.out.println("---- dimension " + dim[idx_dim] + "-D done ----");
        }
    }


 

/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 0 ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */

    /*
    * Generates an array containing the possible values that each coordinate of input vectors can take
    * When generating input vectors, generateRandomInput() will select a random value within the possible values contained in this table
    */
    public static double[] generateKnownInput() {

        //System.out.println("Generating known inputs");

        // the size is calculated in function of the domain width and the chosen interval
        int arraySize = (int)((MAX_RANGE_VALUE - MIN_RANGE_VALUE) / RANGE_INTERVAL + 1);
        double[] res = new double[arraySize];
        //System.out.println("Array size :" + arraySize);

        double j = MIN_RANGE_VALUE;
        
        for(int i = 0, length = res.length; i < length; i++) {
            res[i] = j;
            j += RANGE_INTERVAL;
        }
        /*
        for(int i = 0, length = res.length; i < length; i++) {
            System.out.println(res[i]);
        }

        */

        //System.out.println("-------------------------------------------------\n\n\n");
        
        return res;
    }

/* ************************************************************************************************************************************************ */


    /*
   * Generates the random input vectors that will be given to each chromosome
   */
    public static double[][] generateSampleInputData(int dimension, double[] knownInputValuesTable){

        //System.out.println("Generating input sample data");

        double[][] res = new double[NUMBER_OF_INPUT_VECTORS][dimension];

        for (int i = 0; i < NUMBER_OF_INPUT_VECTORS; i++) {
                res[i] = generateRandomInput(dimension,knownInputValuesTable);
                
        }

        //System.out.println("-------------------------------------------------\n\n\n");

        return res;
    }

/* ************************************************************************************************************************************************ */

    /*
   * Generates one random input vector that will be given to each chromosome
   */
    public static double[] generateRandomInput(int dimension, double[] knownInputValuesTable) {

        //System.out.println("Generating random input");

        double[] res = new double[dimension];

        Random rd = new Random();

        res[dimension-1] = 0;

        int temporary_index = 0;

        for (int i = 0; i < dimension-1; i++) {
            temporary_index = rd.nextInt(knownInputValuesTable.length-1);
            // knownInputValuesTable contains the values that are allowed to give to input vector coordinates
            res[i] = knownInputValuesTable[temporary_index];
            res[dimension-1] += Math.pow(res[i],2);

        }

/*
        for (int i = 0; i < dimension; i++) {
            System.out.println(String.format("%5s", res[i]));
        }
        System.out.println("\n\n");
*/
        //System.out.println("-------------------------------------------------\n\n\n");

        return res;
      
    }




/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 1 ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */

    /*
    * Generate chromosome population randomly
    */
    public static ArrayList<double[]> generatePopulation(int dimension) {

        // this formula is explained in the report
        int sizeOfEachChromosome =  dimension * NUMBER_OF_NEURONS_PER_HIDDEN_LAYER 
                                    + (NUMBER_OF_HIDDEN_LAYERS - 1) * (int)Math.pow(NUMBER_OF_NEURONS_PER_HIDDEN_LAYER,2)
                                    + NUMBER_OF_NEURONS_PER_HIDDEN_LAYER;

        ArrayList<double[]> res = new ArrayList<double[]>();
        
        Random rd = new Random();

        for(int i = 0; i < NUMBER_OF_CHROMOSOMES; i++) {
            
            res.add(new double[sizeOfEachChromosome]);

            for(int j = 0; j < sizeOfEachChromosome; j++) {

                res.get(i)[j] = (rd.nextDouble() - rd.nextDouble());
            }
        }

        return res;

    }



/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 2 and 3 ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */

    /*
    * Generates the MLP output for each chromosome
    */
   
   public static ArrayList<double[]> getMLPPopulationOutput(double[][] sampleInputDataSet, ArrayList<double[]> initialChromosomePopulation) {

        //Each row corresponds to a chromosome. Each column corresponds to an input. 
        //Last Column is the average of all the outputs for a given chromosome (which is why we add 1 extra column)
        //
       ArrayList<double[]> mLPPopulationOutput = new ArrayList<double[]>();
       
    
        for (int i = 0; i < initialChromosomePopulation.size(); i++) {
            double[] mLPOutputTable =  calculateOutputsForChromosome(sampleInputDataSet,initialChromosomePopulation.get(i));
            mLPPopulationOutput.add(mLPOutputTable);

        }

        return mLPPopulationOutput;
    }



/* ************************************************************************************************************************************************ */



    public static double[] calculateOutputsForChromosome(double[][] sampleInputDataSet, double[] chromosome) {

    
        double[] mLPOutput = new double[NUMBER_OF_INPUT_VECTORS+1];

        // -1 because sampleInputDataSet's last column is the output, not an input
        int dimension = sampleInputDataSet[0].length - 1;

        // inputs to first layer
        double [] initialInput = new double[dimension];

        double [] temporaryOuput = new double[NUMBER_OF_NEURONS_PER_HIDDEN_LAYER];

        double temporary = 0;

        // variable to know at what position we are in the chrosome
        int count = 0;


        for(int i = 0; i < NUMBER_OF_INPUT_VECTORS; i++) {

            //System.out.println("INPUT NUMBER " + i);

            mLPOutput[i] = 0;

            count = 0;

            // filling the first input array
            for(int j = 0; j < dimension; j++) {
                initialInput[j] = sampleInputDataSet[i][j];
            }

            // emptying the array at the beginning of each loop
            for(int j = 0; j < NUMBER_OF_NEURONS_PER_HIDDEN_LAYER; j++) {
                temporaryOuput[j] = 0;
            }

            // filling : first layer
            // looping through each neuron in the first layer
            for(int j = 0, len = NUMBER_OF_NEURONS_PER_HIDDEN_LAYER; j < len; j++) {

                temporary = 0;

                // looping through input AND chromosome weights at the same time to calculate sum(wi*xi)
                for (int k = 0; k < dimension; k++  ) {
                    //System.out.println("chromosome k :" + chromosome[count]);
                    temporary += initialInput[k] * chromosome[count];
                    count++;
                }

                //System.out.println("Output of first layer, neuron : " + j + ", value before tanh : " + temporary);
                temporaryOuput[j] = Math.tanh(temporary);
                //System.out.println("Output of first layer, neuron : " + j + ", value : " + temporaryOuput[j]);


            }


            double[] temporaryHiddenLayerInput = temporaryOuput;



            //System.out.println(chromosome.length);

            // filling : hidden layers (size : NUMBER_OF_HIDDEN_LAYERS)
            for(int j = 0; j < NUMBER_OF_HIDDEN_LAYERS -1; j++) {
                //System.out.println("Loop : hidden layer number " + j);

                //System.out.println("THROUGH HIDDEN LAYERS");

                //System.out.println("j :" + j);
                // saving temporary output 
                temporaryHiddenLayerInput = temporaryOuput;
                
                // clearing temporary variable
                temporary = 0;


                for(int k = 0; k < NUMBER_OF_NEURONS_PER_HIDDEN_LAYER; k++) {

                    //System.out.println("k :" + k);

                    // clearing temporary variable
                    temporary = 0;


                    for (int l = 0; l < NUMBER_OF_NEURONS_PER_HIDDEN_LAYER; l++) {

                        //System.out.println("Count : " + count + ", l : " + l);
                        //System.out.println("chromosome l :" + chromosome[count]);
                        temporary += temporaryHiddenLayerInput[l] * chromosome[count];
                        count++;
                    }

                    
                    //System.out.println("Output of first layer, neuron : " + k + ", value before tanh : " + temporary);

                
                    temporaryOuput[k] = Math.tanh(temporary);
                    //System.out.println("Output of first layer, neuron : " + k + ", value : " + temporaryOuput[k]);

                    
                }

                 

                
            }

            //System.out.println("end of hiddden layer loop");

            //filling : last layer (size 1) == OUTPUT

            temporary = 0;

            for (int j = 0; j < NUMBER_OF_NEURONS_PER_HIDDEN_LAYER; j++) {



                //System.out.println("Count : " + count + "NUMBER_OF_NEURONS_PER_HIDDEN_LAYER :" + NUMBER_OF_NEURONS_PER_HIDDEN_LAYER);

                temporary += temporaryOuput[j] * chromosome[count];
                count++;

            }



            //System.out.println("test");

            mLPOutput[i] = Math.tanh(temporary);

            //System.out.println("----------------------------------------------------------------\n\n");
        }


        mLPOutput[NUMBER_OF_INPUT_VECTORS] = getChromosomeFitness(sampleInputDataSet, mLPOutput);


        

        return mLPOutput;
    }

    
/* ************************************************************************************************************************************************ */

    /*
    * Calculates the custom fitness of a chromosome
    */
    // This adds the last column in MLP output table. It will be used to decided which chromosome we kill.
    public static double getChromosomeFitness(double[][] sampleInputDataSet, double [] mLPOutput) {

        double fitness = 0;

        int dimension = (int)sampleInputDataSet[0].length - 1;

         /*System.out.println("sampleInput size :" + sampleInputDataSet[0].length);
            System.out.println("mLPOutput size :" + mLPOutput.length);
            System.out.println("dim :" + dimension);
        */
        for(int i = 0; i < NUMBER_OF_INPUT_VECTORS; i++) {

            //System.out.println("i :" + i);
            // getting the last column of sampleInputDataSet gives us the real output for the given input
            fitness += Math.abs(sampleInputDataSet[i][dimension] - mLPOutput[i]);
        }


        return fitness;

    }


/* ************************************************************************************************************************************************ */


    /*
    * Calculates the custom fitness of a population
    */
    public static double getPopulationFitness(double[][] sampleInputDataSet, ArrayList<double[]> mLPPopulationOutput) {

        double fitness = 0;

        for(int i = 0, length = mLPPopulationOutput.size(); i < length; i++ ) {
            fitness += mLPPopulationOutput.get(i)[NUMBER_OF_INPUT_VECTORS];
        }


        return fitness;

    }






/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 3 only ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */



    /*
    * Main function to generate the next generation : parents selection, crossover, mutation
    */

    public static ArrayList<double[]> generateNextGenerationPopulation(ArrayList<double[]> currentGenerationPopulation, ArrayList<double[]> mLPOutput) {

        //System.out.println("MODIFYING GENERATION : START");
        // cloning current generation
        ArrayList<double[]> modifiedCurrentGeneration = new ArrayList<double[]>();
        for(double[] chromosome : currentGenerationPopulation) {
            modifiedCurrentGeneration.add(chromosome);
        }

        // PARENTS SELECTION
        ArrayList<double[]> parents = new ArrayList<double[]>();
        int[] parentsIndexes = selectParentsIndexesRandomly(modifiedCurrentGeneration); // random parents selection
        //int[] parentsIndexes = selectParentsAsWorstIndividuals(modifiedCurrentGeneration,mLPOutput); //  worst two chromosomes selection
        


        //System.out.println("Indexes : " + parentsIndexes[0] + ",  " + parentsIndexes[1]);
        parents.add(modifiedCurrentGeneration.get(parentsIndexes[0]));
        parents.add(modifiedCurrentGeneration.get(parentsIndexes[1]));


        //displayChromosomes(parents);

        // CROSSOVER 
        parents = performCrossoverFromMiddleIndex(parents); //single point crossover
        //parents = performTwoPointCrossover(parents); // two point crossover

        // MUTATION
        parents = performMutationOfOneGene(parents); // mutation of 1 gene
        //parents = performMutationOfFiveGenes(parents); //mutation of 5 genees

       



        return modifiedCurrentGeneration;

    }



/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 3 : parent selection methods ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */

    /*
    * selects parents randomly within the given population
    */
    public static int[] selectParentsIndexesRandomly( ArrayList<double[]> chromosomePopulation) {

        int[] parentsIndexes = new int[2];

        //System.out.println("Population size :" + chromosomePopulation.size());

        Random rd = new Random();
        int max_index = chromosomePopulation.size()-1;

        //System.out.println("Max index :" + max_index); 

        // need to add 1 because the argument value is excluded
        parentsIndexes[0] = rd.nextInt(max_index+1);
        //System.out.println("Chosen index 1 : " + parentsIndexes[0]);

        int secondParentIndex = -1;
        do{
            // need to add 1 because the argument value is excluded
            secondParentIndex = rd.nextInt(max_index+1);
        }while(secondParentIndex == parentsIndexes[0]);


         

        parentsIndexes[1] = secondParentIndex;

        //System.out.println("Chosen index 2 : " + parentsIndexes[1]);


        return parentsIndexes;

    }


/* ************************************************************************************************************************************************ */

    /*
    * selects the two worst parents within the given population according to custom fitness
    */
    public static int[] selectParentsAsWorstIndividuals(ArrayList<double[]> chromosomePopulation, ArrayList<double[]> givenGenerationOutput) {

        int[] parentsIndexes = new int[2];

        int max_index = chromosomePopulation.size()-1;

        int worstChromosomeIndex = 0;

        for(int i = 0; i < max_index; i++) {

            if(givenGenerationOutput.get(worstChromosomeIndex)[NUMBER_OF_INPUT_VECTORS] < givenGenerationOutput.get(i)[NUMBER_OF_INPUT_VECTORS]) {
                worstChromosomeIndex = i;
            }
        }


        int secondWorstChromosomeIndex = -1;

        if (worstChromosomeIndex == 0) {
            secondWorstChromosomeIndex = 1;
        } else {
            secondWorstChromosomeIndex = 0;
        }

        for(int i = 0; i < max_index; i++) {

            if((givenGenerationOutput.get(secondWorstChromosomeIndex)[NUMBER_OF_INPUT_VECTORS] < givenGenerationOutput.get(i)[NUMBER_OF_INPUT_VECTORS]) && (i != worstChromosomeIndex)) {
                secondWorstChromosomeIndex = i;
            }
        }


        parentsIndexes[0] = worstChromosomeIndex;
        parentsIndexes[1] = secondWorstChromosomeIndex;



        return parentsIndexes;



    }


/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 3 : crossover methods ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */


    /*
    * performs crossover on the two given parents
    * crossover method : single-point crossover, starting from a random index in the second half of the chromosome
    */
    public static ArrayList<double[]> performCrossoverFromMiddleIndex(ArrayList<double[]> parents) {

        ArrayList<double[]> parentsAfterCrossover = new ArrayList<double[]>();
         //System.out.println("CROSSOVER");



        double[] parent1 = parents.get(0);
        double[] parent2 = parents.get(1);

        int middleIndex =  (parent1.length%2 == 0) ? (parent1.length/2 -1) : (parent1.length - 1)/2;

        //System.out.println(middleIndex);

        double[] temporary = new double[parent1.length];
        for(int i = 0; i < parent1.length;i++) {
            temporary[i] = parent1[i];
        }

        //System.out.println("Parent 1 before crossover :");
        //displayOneChromosome(parent1);
        //System.out.println("Parent 2 before crossover:");
        //displayOneChromosome(parent2);


        for(int i = middleIndex; i < parent1.length; i++) {
            parent1[i] = parent2[i];
            parent2[i] = temporary[i];
        }

        //System.out.println("------------------------------------");
        //System.out.println("Parent 1 after crossover:");
         //displayOneChromosome(parent1);

        //System.out.println("Parent 2 after crossover:");
         //displayOneChromosome(parent2);

        //System.out.println("------------------------------------");

        parentsAfterCrossover.add(parent1);
        parentsAfterCrossover.add(parent2);

        return parentsAfterCrossover;
      


    }


/* ************************************************************************************************************************************************ */

    /*
    * performs crossover on the two given parents
    * crossover method : two-point crossover, starting from a random index until a second random index
    */
    public static ArrayList<double[]> performTwoPointCrossover(ArrayList<double[]> parents) {

        ArrayList<double[]> parentsAfterCrossover = new ArrayList<double[]>();
         //System.out.println("CROSSOVER");



        double[] parent1 = parents.get(0);
        double[] parent2 = parents.get(1);

        int index1 = -1, index2 = -1;


         Random rd = new Random();

         index1 = rd.nextInt(parent1.length);

         do{
            index2 = rd.nextInt(parent1.length);

         }while(index1 == index2);

         // this makes sure index1 is smaller than index2
         if(index1 > index2) {
            int tmp = index1;
            index1 = index2;
            index2 = tmp;
         }



        //System.out.println(index1);
         //System.out.println(index2);
    

        //System.out.println("Parent 1 before crossover :");
        //displayOneChromosome(parent1);
        //System.out.println("Parent 2 before crossover:");
        //displayOneChromosome(parent2);

         double temporary = 0;

        for(int i = index1; i < index2; i++) {
            temporary = parent1[i];
            parent1[i] = parent2[i];
            parent2[i] = temporary;
        }

        //System.out.println("------------------------------------");
        //System.out.println("Parent 1 after crossover:");
         //displayOneChromosome(parent1);

       // System.out.println("Parent 2 after crossover:");
        // displayOneChromosome(parent2);

        //System.out.println("------------------------------------");

        parentsAfterCrossover.add(parent1);
        parentsAfterCrossover.add(parent2);

        return parentsAfterCrossover;
      


    }


/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 3 : mutation methods ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */

    /*
    * selects one gene randomly in each parent chromosome and changes it randomly
    */
    public static ArrayList<double[]> performMutationOfOneGene(ArrayList<double[]> parents) {

        ArrayList<double[]> parentsAfterMutation = new ArrayList<double[]>();
        
        //System.out.println("MUTATION");

        double[] parent1 = parents.get(0);
        double[] parent2 = parents.get(1);


        //System.out.println("Parent 1 before mutation :");
        //displayOneChromosome(parent1);
        //System.out.println("Parent 1 before mutation:");
        //displayOneChromosome(parent2);

        Random rd = new Random();

        parent1[rd.nextInt(parent1.length)] = (rd.nextDouble() - rd.nextDouble());
        parent2[rd.nextInt(parent2.length)] = (rd.nextDouble() - rd.nextDouble());

        //System.out.println("------------------------------------");
        //System.out.println("Parent 1 after mutation:");
         //displayOneChromosome(parent1);

        //System.out.println("Parent 2 after mutation:");
         //displayOneChromosome(parent2);

        //System.out.println("------------------------------------");

        parentsAfterMutation.add(parent1);
        parentsAfterMutation.add(parent2);

        return parentsAfterMutation;
      


    }


/* ************************************************************************************************************************************************ */


    /*
    * selects one gene randomly in each parent chromosome and changes it randomly
    */
    public static ArrayList<double[]> performMutationOfFiveGenes(ArrayList<double[]> parents) {

        ArrayList<double[]> parentsAfterMutation = new ArrayList<double[]>();
        
        //System.out.println("MUTATION");

        double[] parent1 = parents.get(0);
        double[] parent2 = parents.get(1);


        //System.out.println("Parent 1 before mutation :");
        //displayOneChromosome(parent1);
        //System.out.println("Parent 1 before mutation:");
        //displayOneChromosome(parent2);

        Random rd = new Random();

        int[] indexes = new int[5];

        indexes[0] = rd.nextInt(parent1.length);

        do{
            indexes[1] = rd.nextInt(parent1.length);

         }while(indexes[1] == indexes[0]);

        do{
            indexes[2] = rd.nextInt(parent1.length);

        }while(indexes[2] == indexes[1] || indexes[2] == indexes[0]);

        do{
            indexes[3] = rd.nextInt(parent1.length);

        }while(indexes[3] == indexes[2] || indexes[3] == indexes[1] || indexes[3] == indexes[0]);

        do{
            indexes[4] = rd.nextInt(parent1.length);

        }while(indexes[4] == indexes[3] || indexes[4] == indexes[2] || indexes[4] == indexes[1] || indexes[4] == indexes[0]);

        //for (int i = 0; i < 5; i++) {
        //    System.out.println(indexes[i]);
        //}

        for (int i = 0; i < 5; i ++) {
            parent1[indexes[i]] = (rd.nextDouble() - rd.nextDouble());
            parent2[indexes[i]] = (rd.nextDouble() - rd.nextDouble());

        }
        
        

        //System.out.println("------------------------------------");
        //System.out.println("Parent 1 after mutation:");
        //displayOneChromosome(parent1);

        //System.out.println("Parent 2 after mutation:");
        //displayOneChromosome(parent2);

        //System.out.println("------------------------------------");

        parentsAfterMutation.add(parent1);
        parentsAfterMutation.add(parent2);

        return parentsAfterMutation;
      


    }

/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** STEP 3 : individual removing method *********************************************************** */
/* ****************************************** *********************************** ******************************************************************* */


    /*
    * Removes the worst individual of a population
    * It is the one that has the highest cumulated error, with error being the absolute of (MLPoutput - function output)
    */

    public static ArrayList<double[]> removeWorstIndividual(ArrayList<double[]> keptGeneration, ArrayList<double[]> modifiedCurrentGenerationOutput) {


            int worstChromosomeIndex = 0;
            
            for (int i = 0, length = modifiedCurrentGenerationOutput.size(); i < length; i++) {

                //System.out.println("Index "+ i + " value :" + modifiedCurrentGenerationOutput.get(i)[NUMBER_OF_INPUT_VECTORS]);
                if(modifiedCurrentGenerationOutput.get(worstChromosomeIndex)[NUMBER_OF_INPUT_VECTORS] < modifiedCurrentGenerationOutput.get(i)[NUMBER_OF_INPUT_VECTORS]) {
                    worstChromosomeIndex = i;
                }

            }


            //System.out.println("Worst chromosome index is " + worstChromosomeIndex );
            //System.out.println("Its value is :" + modifiedCurrentGenerationOutput.get(worstChromosomeIndex)[NUMBER_OF_INPUT_VECTORS]);

            //System.out.println("Generation size :" + keptGeneration.size());
            keptGeneration.remove(worstChromosomeIndex);

            return keptGeneration;

    }




/* ************************************************************************************************************************************************ */


    /*
    * Removes a random individual of the population
    */

    public static ArrayList<double[]> removeRandomIndividual(ArrayList<double[]> keptGeneration) {


            int index = 0;

            Random rd = new Random();
            index = rd.nextInt(keptGeneration.size());
            
            keptGeneration.remove(index);

            return keptGeneration;

    }

/* ****************************************** *********************************** ******************************************************************* */
/* ************************************************** DISPLAY FUNCTIONS ******************************************************************* */
/* ****************************************** *********************************** ******************************************************************* */


    /*
   * Displays a chromosome population with weights having 2 decimal places
   */
   public static void displayChromosomes(ArrayList<double[]> chromosomePopulation) {


    String line = "";


        for(int i = 0, length = chromosomePopulation.size(); i < length; i++) {

            //System.out.println("Chromosome number : " + i);

            

            for(int j = 0, secondLength = chromosomePopulation.get(0).length; j < secondLength; j++ ) {

                    //System.out.println(String.format("%-10s", initialChromosomePopulation[i][j]));
                
                line += String.format("%-7s",doubleFormatter.format(chromosomePopulation.get(i)[j]));

            }

            System.out.println(line);
            line = "";
        }

   }

/* ************************************************************************************************************************************************ */


   /*
   * Displays a chromosome with weights having 2 decimal places
   */
   public static void displayOneChromosome(double[] chromosome) {


            String line = "";

            for(int j = 0, secondLength = chromosome.length; j < secondLength; j++ ) {

                line += String.format("%-7s",doubleFormatter.format(chromosome[j]));

            }

            System.out.println(line);
        
   }


/* ************************************************************************************************************************************************ */


   /*
   * Displays a MLP input and output array for each chromosome of the population
   */
    public static void displayChromosomePopulationOutput(ArrayList<double[]> mLPPopulationOutput) {

        //int count = 1;


        for(double[] outputs : mLPPopulationOutput) {

            //System.out.println("Output of chromosome number " + count);

            for(int i = 0, len = outputs.length; i < len; i++) {

                System.out.println(outputs[i]);
            }

            //count++;
        }

   }



}

    