package IDPC_DU.Multifactorial_Evolutionary_Algorithm.IDPC_DU_EDGES1;

import java.util.Random;
import java.util.Scanner;

public class Mfea4IDPCDU {
    public static Random random = new Random(0);
    Scanner sc = new Scanner(System.in);
    public int inputTask;
    public static Mfea4IDPCDU instance;

    public static void main(String [] args) {
    	instance=new Mfea4IDPCDU();
        instance.readData();
        instance.init();
        
    }
    void init() {
    	Population pop=new Population(inputTask);
        pop.Evolution();
    }
    
    void readData() {
    	System.out.print("Input the number of task: ");
        inputTask = sc.nextInt();

        //Giai doan tien xu ly
        Files.CreateData(inputTask);
        Files.ReadFile(inputTask);
        //Loc canh
        for (int i = 0; i < Files.TASK_NUM; i++) {
            PreProcess.EdgeFilter(Files.edges[i]);
        }

        //Tao khong gian tim kiem dong nhat
        PreProcess.CreateUSS(Files.edges);

        //Tach rieng cac canh cua tung node
        for (int i = 0; i < Files.TASK_NUM; i++) {
            PreProcess.SeperateData(Files.edges[i], Files.edgesTask.get(i));
        }

        // for (int j = 0; j < Files.edgesTask.get(0)[1].length; j++) {
        //     for (int j2 = 0; j2 < Files.edgesTask.get(0)[0][j].size(); j2++) {
        //         System.out.print(" src " + Files.edgesTask.get(0)[0][j].get(j2).src + " des " +Files.edgesTask.get(0)[0][j].get(j2).des + " weight " + Files.edgesTask.get(0)[0][j].get(j2).weight+ " domain " + Files.edgesTask.get(0)[0][j].get(j2).domain);
        //     }
        //     System.out.println(" ");
        // }

    }
}
