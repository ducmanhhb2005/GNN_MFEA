package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Population {
	public static int curTask=0;
	public int REITERATE_TIME=500;
	public int [] LowestCost;
    public int [] time;
    public int POP_SIZE = 100;    
    public float crossoverRate=0.5f;
    
    public double totalFitness = 0;
    public List<Circle> population = new ArrayList<Circle>();
    public List<Circle> selectedList = new ArrayList<Circle>();
    public List<Circle> childList = new ArrayList<Circle>();
    
    Population(int numtask){
    	LowestCost=new int[numtask];
        time=new int[numtask];
        resetCost();
    }

    void resetCost(){
        for (int i = 0; i < LowestCost.length; i++) {
            LowestCost[i] = 99999;
            time[i]=0;
        }
    }

    public void InitPopulation(){
        population.clear();
        for (int i = 0; i < POP_SIZE; i++) {
            Circle c = new Circle(PreProcess.sample_node, PreProcess.css_node_out_edge);
            population.add(c);
        }

        //Calculate the factorial cost for each element
        for (Circle c : population) {
            for (int i = 0; i < Files.TASK_NUM; i++) {
                c.CalculateFactorialCost(i);
            }
        }
    }

    public void Evaluation(){
        //Find the Factorial Rank for Each Task 
    	curTask = 0;
        for (int i = 0; i < Files.TASK_NUM; i++) {
            Collections.sort(population, Circle.TaskCostComp);
            for (Circle c : population) {
                c.f_rank[i] = population.indexOf(c) + 1;
            }
            curTask++;
        }

        for (Circle c : population) {
            Circle.ScalarFitnessCalculation(c);
        }

        for (Circle c : population) {
            c.best_fit = Circle.BestFitEvaluation(c);
            c.s_factor = Circle.SkillFactorAssigning(c.s_fit);
        }

        Collections.sort(population, Circle.BestFitComp);

        //Calculate the total fitness of the population
        totalFitness = 0.0f;
        for (int i = 0; i < population.size(); i++) {
            totalFitness += population.get(i).best_fit;
        }
    }

    public void Selection() {
        selectedList.clear();
        int parentForGO = (int)(POP_SIZE*crossoverRate);
        
      //Keep the best fitness individuals of each task
        // for (int i = 0; i < Files.TASK_NUM; i++) {
	    //     for (int j = 0; j < population.size(); j++) {
	    //         if(population.get(j).f_rank[i] == 1){
	    //         	selectedList.add(population.get(j));
	    //             break;
	    //         }
	    //     }
        // }
        //Get total fitness of the population -Files.TASK_NUM
        for (int i = 0; i < parentForGO; i++) {
        	selectedList.add(Roulette());
        }
    }
    
    public List<Circle> Reproduce() {
    	int rand1;
        int rand2;
    	int childNum = POP_SIZE;
        childList.clear();
        while(childNum > 0) {
            rand1 = Mfea4IDPCDU.random.nextInt(selectedList.size());
            rand2 = Mfea4IDPCDU.random.nextInt(selectedList.size());
            
            if(childList.addAll(GeneticMechanism.GeneticOperators(selectedList.get(rand1), selectedList.get(rand2)))) {
            	childNum -= 2;
            }
        }
    
        return childList;
    }

    public Circle Roulette() {
        double value = Mfea4IDPCDU.random.nextDouble() ;

        double sum = 0;
        int position=0;

        for (int i = 0; i < population.size(); i++) {
            sum += population.get(i).best_fit;
            if (value <= sum/totalFitness) {
            	position=i;
                break;
            }
        }
        return (population.get(position));
    }
    
    public void Evolution() {
        
        for (int h = 0; h < 30; h++) {
            resetCost();
            //Khoi tao quan the
            InitPopulation();
            //Danh gia quan the ban dau
            Evaluation();
            for (int i = 0; i < REITERATE_TIME; i++) {
                Selection();
                population = Reproduce();
                Evaluation();
                
                for (int j = 0; j < Files.TASK_NUM; j++) {
                    for (int k = 0; k < population.size(); k++) {
                        if(population.get(k).f_rank[j] == 1){
                            if (population.get(k).f_cost[j] < LowestCost[j]){
                                LowestCost[j] = population.get(k).f_cost[j];
                                time[j] = i;
                                break;
                            }
                        }
                    }
                }
            }
        }
        
        for (Circle c : population) {       
            for (int i = 0; i < c.nodePri.length; i++) {
                System.out.print(c.nodePri[i] + " ");
            }
            System.out.println(" ");
            for (int i = 0; i < c.edgeIdx.length; i++) {
                System.out.print(c.edgeIdx[i] + " ");
            }
            System.out.println(" ");

            for (int i = 0; i < c.f_rank.length; i++) {
                System.out.print(c.f_cost[i]+ " " + c.f_rank[i]+ " " );
            }
            System.out.println(" ");
        }

        for (int i = 0; i < LowestCost.length; i++) {
                System.out.print("Lowest Cost Task " + i + ": " + LowestCost[i] + " Time " + i + ": "+ time[i]+ " ---- ");
            }
            System.out.println(" ");
    }
}
