from anytree.importer import DictImporter
from anytree.exporter import UniqueDotExporter
from IPython.display import Image

class Tree:
    def __init__(self,max_items_per_turn):
        self.max_items_per_turn=max_items_per_turn

    #   input: object_state (number of remaining objects)
    #   output: all candidates that can be a new state
    def state_candidates(self,object_state):
        return [object_state - num_remove for num_remove in range(1,self.max_items_per_turn+1) if num_remove <= object_state]


    #   Each node has 'object_state', 'current_player', 'score' attributes
    #   If you specify a value for the 'children' attribute, it will be added as a child node.
    def construct_tree(self,object_state,current_player):
        tree_dict={}
        if current_player=="Max":  #"Max" indicates the maximizer while "Min" indicates the minimizer.
            tree_dict['object_state']=object_state
            tree_dict['current_player']=current_player
            tree_dict['score']=None
            tree_dict['children']=[
                self.construct_tree(new_state,"Min")
                for new_state in self.state_candidates(object_state)
            ]
            return tree_dict

        else:
            tree_dict['object_state']=object_state
            tree_dict['current_player']=current_player
            tree_dict['score']=None
            tree_dict['children']=[
                self.construct_tree(new_state,"Max")
                for new_state in self.state_candidates(object_state)
            ]
            return tree_dict
    

    def convert_dict_to_tree(self,tree_dict):
        importer = DictImporter()
        return importer.import_(tree_dict)


    #visualize tree
    def export_tree(self,node,is_score_tree,tree_name):
        if is_score_tree:
            UniqueDotExporter(node,
            nodeattrfunc=lambda n: 'label="%s:%s"' % (n.current_player,n.object_state),
            edgeattrfunc=lambda n,c:'label="%s"' %(c.score)
            ).to_picture(tree_name+".png")
        else:
            node=self.convert_dict_to_tree(node)
            UniqueDotExporter(node,
                              nodeattrfunc=lambda n: 'label="%s:%s"' % (n.current_player,n.object_state)
                              ).to_picture(tree_name+".png")
            


class Minimax: 
    def score_leaves(self):
        for leaf in self.root.leaves:
            if leaf.current_player=="Max":
                leaf.score=-1
            else:
                leaf.score=1

    def check_leaf(self,node):
        if node.is_leaf:
            return node.score

    def minimax(self,node,current_player):
        scores=[]
        if (score := self.check_leaf(node)) is not None:
            return score

        if node.current_player=="Max":
            '''   
            [TODO 1]
            Get the children of "node".
            Then, get the children's scores.
            Update the value of node.score to the appropriate score.(to visualize the tree)
            Return appropriate value given the player.

            * The value of node.score and return value should be same
            HINT:
            1. You may use "node.children" which returns the children of node
            2. You may call "minimax" function recursively.
            '''
            maxScore = float('-inf')

            for c in node.children:
                c.score = self.minimax(c, 'Min')  # min layer
                maxScore = max(maxScore, c.score)

            return maxScore
        else:
            '''   
            [TODO 2]
            Get the children of "node".
            Then, get the children's scores.
            Update the value of node.score to the appropriate score.(to visualize the tree)
            Return appropriate value given the player.

            * The value of node.score and return value should be same

            HINT:
            1. The code will be similar to the above [TODO].

            '''
            minScore = float('inf')

            for c in node.children:
                c.score = self.minimax(c, 'Max')  # max layer
                minScore = min(minScore, c.score)

            return minScore 

class Nim(Minimax): # Why Nim takes Minimax? What's the benefit? => You can refer to "Appendix - Python Inheritance" section in this notebook.
    def __init__(self, max_items_per_turn,root,initial_object,user_first):
        super().__init__() # Get fucntions and properties of Minimax. Now, you can call them using "self.". (e.g., self.score_leaves())
        self.root=root
        self.initial_object=initial_object
        self.user_first=user_first

    def max_action(self,node):
        scores=[]
        new_states=[]

        '''   
        [TODO 3]
        Get the children of "node".
        Get a score of each child.

        HINT:
        1. You may use "node.children" which returns the children of node
        2. You may call "minimax" function to get a score of each child.
        '''
        for c in node.children:
            score = self.minimax(c, c.current_player)
            scores.append(score)
            c.score = score
        #lambda x : returns scores[x]
        return node.children[max(range(len(scores)), key = lambda x : scores[x])]


    def user_input(self,node):
        num_remove=int(input("Enter the number of objects to remove:"))
        for i in node.children:
            if i.object_state==node.object_state-num_remove:
                return i
        return node

    def game_manager(self):
        #Score leaf nodes before starting the game
        self.score_leaves()
        current_node=self.root

        for i in range(1,self.initial_object+1):
            if i==1:print(f"Initial Object:{self.initial_object}")
            #User turn
            if (i+self.user_first)%2==0:
                old_node=current_node
                #keep requesting user input until the user enters an appropriate value
                while old_node==current_node:
                    current_node=self.user_input(current_node)
                    if old_node==current_node:
                        print("wrong input!")
                    else:
                        print(f"USER | Removed {old_node.object_state-current_node.object_state}, Remaining objects: {current_node.object_state}")

                if current_node.object_state==0:
                    print('User win')
                    break

            #Program turn
            else:
                old_node=current_node
                current_node=self.max_action(current_node)
                print(f"PROGRAM | Removed {old_node.object_state-current_node.object_state}, Remaining objects: {current_node.object_state}")
                
                if current_node.object_state==0:
                    print('Program win')
                    break




class Minimax_pruning(Minimax):
    def __init__(self,root):
        super().__init__() # get the functions and properties of parent class(Minimax)
        self.root=root

    #   We set alpha and beta to -1 and 1, respectively instead of -inf and inf
    def minimax_pruning(self, node, current_player, alpha=-1, beta=1):
        if (score := self.check_leaf(node)) is not None:
            return score

        scores = []
        for child in node.children:
            if current_player=="Max":
                '''   
                [TODO 1]
                Get the scores of child and update alpha
                HINT:
                1. The code can be similar to "minimax" function in Task1

                '''
                maxScore = float('-inf')
                
                for c in node.children:
                    c.score = self.minimax_pruning(c, 'Min', alpha, beta)
                    maxScore = max(maxScore, c.score)
                    alpha = max(alpha, maxScore)
                    if alpha >= beta:
                        break

                return maxScore 


            else:
                '''   
                [TODO 2]
                Get the scores of child and update beta
                HINT:
                1. The code can be similar to "minimax" function in Task1 and the above [TODO]

                '''
                minScore = float('inf')

                for c in node.children:
                    c.score = self.minimax_pruning(c, 'Max', alpha, beta)
                    minScore = min(minScore, c.score)
                    beta = min(beta, minScore)
                    if alpha >= beta:
                        break

                return minScore 

            '''   
            [TODO 3]
            Prune the tree according to the alpha and beta.
            Update the value of node.score to the appropriate score.(to visualize the tree)
            Return the appropriate score.

            * The value of node.score and return value should be same
            '''
