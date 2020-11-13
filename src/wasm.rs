use crate::expectimax::Expectimax;
use crate::engine as GameEngine;
use crate::engine::{Move, Board};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmExpectimax;

#[wasm_bindgen]
impl WasmExpectimax {
    pub fn new() -> Self {
        GameEngine::new();
        Expectimax::new();
        WasmExpectimax
    }

    //pub fn get_next_move(&self, board: Board) -> i32 {
    //    match Expectimax.get_next_move(board) {
    //        Some(direction) => match direction {
    //            Move::Up => 0,
    //            Move::Right => 1,
    //            Move::Down => 2,
    //            Move::Left => 3,
    //        },
    //        None => self.get_possible_move(board),
    //    }
    //}

    fn get_possible_move(&self, board: Board) -> i32 {
        for &direction in &[Move::Up, Move::Down, Move::Left, Move::Right] {
            let new_board = GameEngine::shift(board, direction);
            if new_board != board {
                match direction {
                    Move::Up => return 0,
                    Move::Right => return 1,
                    Move::Down => return 2,
                    Move::Left => return 3,
                }
            }
        }
        return -1;
    }
}
