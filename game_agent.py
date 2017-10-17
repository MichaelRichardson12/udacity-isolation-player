"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def common_moves(game, player):
    pmoves = game.get_legal_moves()
    omoves = game.get_legal_moves(game.get_opponent(player))
    cmoves = pmoves and omoves
    return 8 - len(cmoves)


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    opp = game.get_opponent(player)
    player_moves = game.get_legal_moves()
    opponent_moves = game.get_legal_moves(opp)

    common_moves = player_moves and opponent_moves

    unique_player_moves = list(set(player_moves)^set(common_moves))
    unique_opponent_moves = list(set(opponent_moves)^set(common_moves))

    n_own_moves_left = len(unique_player_moves)
    n_opp_moves_left = len(unique_opponent_moves)

    return float(n_own_moves_left - n_opp_moves_left)


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    opp = game.get_opponent(player)
    player_moves = game.get_legal_moves()
    opponent_moves = game.get_legal_moves(opp)

    common_moves = player_moves and opponent_moves

    factor = 1 / (game.move_count + 1)

    unique_player_moves = list(set(player_moves)^set(common_moves))
    unique_opponent_moves = list(set(opponent_moves)^set(common_moves))

    n_own_moves_left = len(unique_player_moves)
    n_opp_moves_left = len(unique_opponent_moves)

    return float(n_own_moves_left - factor * n_opp_moves_left)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    opp = game.get_opponent(player)
    player_moves = game.get_legal_moves()
    opponent_moves = game.get_legal_moves(opp)

    common_moves = player_moves and opponent_moves

    factor = 1 / (game.move_count + 1)

    unique_player_moves = list(set(player_moves)^set(common_moves))
    unique_opponent_moves = list(set(opponent_moves)^set(common_moves))

    n_own_moves_left = len(unique_player_moves)
    n_opp_moves_left = len(unique_opponent_moves)

    return float((1 - factor) * n_own_moves_left - factor * n_opp_moves_left)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        return self.min_max_recustion_function(game, depth)[0]

    def get_active_player(self, game):
        """
        Get the current active user
        """
        return game.active_player == self

    def min_max_recustion_function(self, game, depth):
        """
        just a wrapper of the old code logic returning a tuple (move, score)
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # Return players position and score if depth is equal to zero
        if depth == 0:
            return (game.get_player_location(self), self.score(game, self))

        utility_value, min_max_func, best_move = None, None, (-1, -1)

        if self.get_active_player(game):
            # Set function to use as max if it is the max calculation
            min_max_func, utility_value = max, float("-inf")
        else:
            # Set function to use as min if it is the min calculation
            min_max_func, utility_value = min, float("inf")

        # Loop through possible moves
        for move in game.get_legal_moves():
            # Get the result of a move
            new_board = game.forecast_move(move)

            # Recursion set that goes until the depth is equal to zero
            new_score = self.min_max_recustion_function(new_board, depth - 1)[1]

            if min_max_func(utility_value, new_score) == new_score:
                best_move = move
                utility_value = new_score

        return (best_move, utility_value)


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0:
            return (-1, -1)

        # Select a random move in case the search fails from a timeout
        best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        try:
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            # Handel time exceeded exceptions
            pass

        # Return the most recently updated best move
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.
        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md
        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************
        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state
        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting
        alpha : float
            Alpha limits the lower bound of search on minimizing layers
        beta : float
            Beta limits the upper bound of search on maximizing layers
        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves
        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.
            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        active_player = game._active_player

        def _max_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # If current player is a winner or loser, return the score
            if game.is_winner(self) or game.is_loser(self):
                return self.score(game, active_player)

            # Return score if depth is zero
            if depth == 0:
                return self.score(game, active_player)

            # Set the initialvalue to negative infinity
            value = float("-inf")

            for move in game.get_legal_moves():
                # Get the result of a move
                new_board = game.forecast_move(move)

                # Find the max between the value and recursive _min_value
                value = max(value, _min_value(new_board, depth-1, alpha, beta))

                if value >= beta:
                    return value

                # Set alpha to the max between the two values
                alpha = max(alpha, value)

            return value

        def _min_value(game, depth, alpha, beta):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()

            # If current player is a winner or loser, return the value of the utility function
            if game.is_winner(self) or game.is_loser(self):
                return game.utility(active_player)

            # Return score if depth is zero
            if depth == 0:
                return self.score(game, active_player)

            # Set the initialvalue to positive infinity
            value = float("inf")

            for move in game.get_legal_moves():
                # Get the result of a move
                new_board = game.forecast_move(move)

                # Find the max between the value and recursive _max_value
                value = min(value, _max_value(new_board, depth-1, alpha, beta))

                if value <= alpha:
                    return value

                # Set beta to the max between the two values
                beta = min(beta, value)

            return value

        # Initialize the values
        best_value, best_move = float("-inf"), (-1, -1)

        for move in game.get_legal_moves():
            # Get the result of a move
            new_board = game.forecast_move(move)

            # Find the max between the value and recursive _min_value
            value = _min_value(new_board, depth-1, alpha, beta)

            # Set alpha to the max between the two values
            alpha = max(alpha, value)

            # If the new value is gretater than the current best value set the the best move to the new move
            if value > best_value:
                best_value = value
                best_move = move

        # Return the best move
        return best_move
