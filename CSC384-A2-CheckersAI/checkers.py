import argparse
import copy

# ===============================================================================================================
                                            # GLOBAL VARIABLES
# ===============================================================================================================
simple_moves = [(1,1), # bottom right
                (1,-1), # bottom left
                (-1,1), # top right
                (-1,-1) # top left
                ]
jump_moves = [(2,2), # bottom right
              (2,-2), # bottom left
              (-2,2), # top right
              (-2,-2) # top left
                ]

red_moves = simple_moves[2:]
red_jump_moves = jump_moves[2:]
black_moves = simple_moves[:2] 
black_jump_moves = jump_moves[:2]
king_moves = simple_moves
king_jump_moves = jump_moves

cache = {} 

# ===============================================================================================================
                                            # GLOBAL HELPER FUNCTIONS
# ===============================================================================================================
def initialize_board(state):
    '''Function to initialize board pieces to objects'''
    for i in range(8):
        for j in range(8):
            piece = state.board_arr[i][j]
            position = (i, j)
            if piece in ['r','R']:
                state.board_dict[position] = piece
                state.redCount += 1
                state.piecesCount += 1

            elif piece in ['b','B']:
                state.board_dict[position] = piece
                state.blackCount += 1
                state.piecesCount += 1
    print(state)

def isRed(piece):
    '''Returns a boolean if piece is red or not'''
    if piece in ['r', 'R']:
        return True
    else:
        return False
    
def isKing(piece):
    '''Returns a boolean if piece is king or not'''
    if piece in ['B', 'R']:
        return True
    else:
        return False
    
def get_player(piece):
    '''Returns which player piece belongs to'''
    if isRed(piece):
        return 'r'
    else:
        return 'b'

def get_opp_turn(piece):
    '''Returns opponent player's color'''
    if isRed(piece):
        return 'b'
    else:
        return 'r'
    
def read_from_file(filename):
    f = open(filename)
    lines = f.readlines()
    board = [[str(x) for x in l.rstrip()] for l in lines]
    f.close()

    return board

# ===============================================================================================================
                                            # STATE CLASS
# ===============================================================================================================
class State:
    def __init__(self, board):
        self.width = 8
        self.height = 8
        self.redCount = 0
        self.blackCount = 0
        self.piecesCount = 0

        self.board_arr = board
        self.board_dict = {}
    
    def __str__(self):
        '''Print board'''
        array = self.update_board_arr()
        s = ''
        for i in array:
            for j in i:
                s += str(j)
            s += '\n'
        return s
    
    def update_board_arr(self):
        '''Helper function to update board arr so that it can be printed'''
        a = [['.' for i in range(8)] for j in range(8)]
        for pos, piece in self.board_dict.items():
            i, j = pos
            a[i][j] = piece

        return a


    ################# helper functions #################
    def is_empty(self, new_pos):
        '''Helper function to check if position is occupied. Returns boolean.'''
        if new_pos in self.board_dict:
            return False
        else:
            return True
        
    def is_valid_pos(self, new_pos):
        '''Helper function to check if position is out of bounds.
            Returns a boolean if it's within bounds of the board.'''
        i, j = new_pos
        return (0 <= i <= 7) and (0 <= j <= 7)
        
    def is_valid_move(self, curr_pos, new_pos):
        '''Helper function to check if new position is occupied, within bounds
            and if new position is valid based on checkers rules.
            Returns a boolean on whether the specified move is valid'''
        curr_i, curr_j = curr_pos
        piece = self.board_dict[curr_pos]
        new_i, new_j = new_pos

        if self.is_empty(new_pos):
            if self.is_valid_pos(new_pos):
                move_i = new_i - curr_i
                move_j = new_j - curr_j
                
                # jump
                if abs(move_i) > 1 and abs(move_j) > 1:
                    # check to see if the piece capture is of the opp turn
                    if move_i > 0:
                        opp_i = curr_i + move_i - 1
                    else:
                        opp_i = curr_i + move_i + 1

                    if move_j > 0:
                        opp_j = curr_j + move_j - 1
                    else:
                        opp_j = curr_j + move_j + 1

                    opp_pos = (opp_i, opp_j)

                    if opp_pos not in self.board_dict:
                        return False
                
                    else:
                        opp_piece = self.board_dict[opp_pos]
                        if isRed(piece) == isRed(opp_piece):
                            return False
            else:
                return False
        else:
            return False
                    
        return True

    def is_terminal(self):
        '''Helper function to check if game has ended.
            Returns boolean if game has ended.'''
        return len(self.board_dict) == 1    
    
    def king_upgrade(self, pos, piece):
        '''Helper function to check if piece can be upgraded'''
        i, j  = pos

        if not isKing(piece):
            if isRed(piece) and i == 0:
                return True
            elif not isRed(piece) and i == 7:
                return True
            
        return False
            

    ################# evaluation functions #################
    def evaluate_piece(self, curr_pos, piece):
        '''Function that evaluates score of current piece based on position on the board.
            Returns an integer.'''
        i, _ = curr_pos
        if isRed(piece):
            return 28-i*4
        else:
            return i*4
        
    def evaluate_state(self):
        '''Function to evaluate score. Returns an integer'''
        score = 0
        for curr_pos, piece in self.board_dict.items():
            if isRed(piece):
                if isKing(piece):
                    score += 5
                else:
                    score += self.evaluate_piece(curr_pos, piece)
                    score += 3
            
            else:
                if isKing(piece):
                    score -= 5
                else:
                    score -= self.evaluate_piece(curr_pos, piece)
                    score -= 3

            i, j = curr_pos
            neighboring_pos = [(i+1, j+1), (i-1, j+1), (i+1, j-1), (i-1, j-1)]

            for pos in neighboring_pos:
                if pos in self.board_dict and isRed(self.board_dict[pos]) == isRed(piece):
                    if isRed(piece):
                        score += 2
                    else:
                        score -= 2
            if self.redCount > self.blackCount:
                score += (24-self.redCount)*2
            elif self.blackCount > self.redCount:
                score -= (24-self.blackCount)*2
            
        return score
    

    ################# move functions #################
    def move(self, curr_pos, new_pos):
        '''Function to make move. Returns a State object'''
        curr_i, curr_j = curr_pos
        new_i, new_j = new_pos

        board_copy = copy.deepcopy(self)
        piece = board_copy.board_dict[curr_pos]

        if board_copy.is_valid_move(curr_pos, new_pos):
            move_i = new_i - curr_i
            move_j = new_j - curr_j
                
            if abs(move_i) > 1 and abs(move_j) > 1:
                if move_i > 0:
                    opp_i = curr_i + move_i - 1
                else:
                    opp_i = curr_i + move_i + 1

                if move_j > 0:
                    opp_j = curr_j + move_j - 1
                else:
                    opp_j = curr_j + move_j + 1

                opp_pos = (opp_i, opp_j)
                
                if isRed(board_copy.board_dict[opp_pos]):
                    board_copy.redCount -= 1
                else:
                    board_copy.blackCount -= 1
                board_copy.piecesCount -= 1

                del board_copy.board_dict[opp_pos]
            
            if self.king_upgrade(new_pos, piece):
                piece = piece.upper()

            board_copy.board_dict[new_pos] = piece
            del board_copy.board_dict[curr_pos]

        else:
            return board_copy

        return board_copy
    
    def jump_simple_moves(self, curr_pos, piece, jump=True):
        '''Helper function to generate all jump and simply moves. Returns an array '''
        moves = []
        curr_i, curr_j = curr_pos
        moveset = None

        if jump:
            if isKing(piece):
                moveset = king_jump_moves
            else:
                if isRed(piece):
                    moveset = red_jump_moves
                else:
                    moveset = black_jump_moves
        else:
            if isKing(piece):
                moveset = king_moves
            else:
                if isRed(piece):
                    moveset = red_moves
                else:
                    moveset = black_moves

        for move in moveset:
            m_i, m_j = move
            n_i = curr_i + m_i
            n_j = curr_j + m_j
            new_pos = (n_i, n_j)

            if self.is_valid_move(curr_pos, new_pos):
                moves.append(new_pos)

        return moves

    def get_all_moves(self, turn):
        '''Helper function to get all possible simple and 
            jump moves (including multi-jumps) for current state.
            Returns two dictionaries with current position as key and all new positions as value.'''
        simple_dict = {}
        jump_dict = {}

        for curr_pos, piece in self.board_dict.items():
            if get_player(piece) == turn:
                # checking if there's a jump available
                jump_moves = self.jump_simple_moves(curr_pos, piece)
                
                # if there's no jump moves, then we consider simple moves
                if not jump_moves:
                    simple_moves = self.jump_simple_moves(curr_pos, piece, False)
                    simple_dict[curr_pos] = simple_moves
                
                # there are jump moves
                else:
                    jump_dict[curr_pos] = {curr_pos: jump_moves} 
                    f_pos = curr_pos
                    new_board = copy.deepcopy(self)

                    while jump_moves:
                        for pos in jump_moves:
                            new_board = new_board.move(f_pos, pos)
                            next_jumps = new_board.jump_simple_moves(pos, piece)

                            if next_jumps:
                                jump_dict[curr_pos][pos] = next_jumps
                                
                            f_pos = pos
                        
                        jump_moves = next_jumps
        
        return simple_dict, jump_dict
    
    ################# successor states #################
    def multi_jump_successors(self, jump_dict, out=[]):
        '''Helper function that recursively generates all successors from multi-jump sequences.
            Returns a list of successor states from multi-jumps.'''
        successors = []
        for prev_pos, new in jump_dict.items():
            for p in new:
                if p in jump_dict:
                    if prev_pos in self.board_dict:
                        i_state = self.move(prev_pos, p)
                        jump_dict = {i:jump_dict[i] for i in jump_dict if i!=prev_pos}
                        s = i_state.multi_jump_successors(jump_dict, successors)
                        successors += s

                else:
                    if prev_pos in self.board_dict:
                        f_state = self.move(prev_pos, p)
                        successors.append(f_state)
        
        return successors

    def get_all_successors(self, turn):
        '''Function that returns all possible states considering only valid jumps. 
            If there are jump_moves to be made, simple moves would be ignored
            Returns a list of successor State objects.'''
        successors = []
        simple, jump = self.get_all_moves(turn)

        # if there is a jump successor state, we prioritize it and ignore all simple moves
        if jump:
            for prev_pos, new_pos_ds in jump.items():
                    s = self.multi_jump_successors(new_pos_ds)
                    successors += s

        else:
            for prev_pos, new_pos in simple.items():
                for p in new_pos:
                    f_state = self.move(prev_pos, p)
                    successors.append(f_state)
        
        return successors

# ===============================================================================================================
                                            # ALPHA BETA PRUNING
# ===============================================================================================================
def max_value(turn, state, alpha, beta, depth=9):
    if cut_off_test(state, depth):
        return None, state.evaluate_state()
    
    v = -float('inf')
    best_state = None
    successor_states = state.get_all_successors(turn)
    successor_states.sort(key=lambda state:state.evaluate_state(), reverse=True)

    for s_state in successor_states:
        if s_state.__str__() not in cache:
            i_state, new_v = min_value('b', s_state, alpha, beta, depth-1)
            if new_v > v:
                v = new_v
                best_state = s_state
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        else:
            v = cache[s_state.__str__()]
            best_state = s_state
       
    cache[best_state.__str__()] = v

    return best_state, v

def min_value(turn, state, alpha, beta, depth=9):
    if cut_off_test(state, depth):
        return None, state.evaluate_state()
    
    v = float('inf')
    successor_states = state.get_all_successors(turn)
    successor_states.sort(key=lambda state:state.evaluate_state())
    best_state = None

    for s_state in successor_states:
        if s_state.__str__() not in cache:
            i_state, new_v = max_value('r', s_state, alpha, beta, depth-1)
            if new_v < v:
                v = new_v
                best_state = s_state
            beta = min(beta, v)
            if beta <= alpha:
                break
        else:
            v = cache[s_state.__str__()]
            best_state = s_state

    cache[best_state.__str__()] = v

    return best_state, v

def cut_off_test(state, depth):
    return depth==0 or state.is_terminal()

# ===============================================================================================================
                                            # MAIN ()
# ===============================================================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzles."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    args = parser.parse_args()

    initial_board = read_from_file(args.inputfile)
    state = State(initial_board)
    initialize_board(state)

    turn = 'r'
    f = open(args.outputfile, 'w')
    f.write(state.__str__())
    f.write('\n')

    while state != None:
        if turn == 'r':
            state, v = max_value(turn, state, -float('inf'), float('inf'))
            print(turn, state)
            turn = 'b'
        
        else:
            state, v = min_value(turn, state, -float('inf'), float('inf'))
            print(turn, state)
            turn = 'r'
            
        if state != None:
            f.write(state.__str__())
            f.write('\n')

    f.close()

    print(cache)