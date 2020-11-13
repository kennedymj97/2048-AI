use ai_2048::engine as GameEngine;
use ai_2048::engine::{Board, Move};
use ai_2048::expectimax::{Expectimax, ExpectimaxMultithread};
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    //println!("Creating file...");
    //let path = Path::new("./results2.csv");
    //let mut f = File::create(path).expect("Failed to create file");
    //f.write("score,highest_tile\n".as_bytes()).expect("failed to write column headers to file");
    //for run in 0..50 {
    //    println!("Running game {}...", run + 1);
    //    GameEngine::new();
    //    let mut expectimax = ExpectimaxMultithread::new();
    //    let mut board = GameEngine::insert_random_tile(0);
    //    board = GameEngine::insert_random_tile(board);
    //    while !GameEngine::is_game_over(board) {
    //        let direction = expectimax.get_next_move(board);
    //        if direction.is_none() {
    //            break;
    //        }
    //        board = GameEngine::make_move(board, direction.unwrap());
    //    }
    //    f.write_fmt(format_args!("{},{}\n", GameEngine::get_score(board), GameEngine::get_highest_tile_val(board))).expect("failed to write to file");
    //}
    GameEngine::new();
    let mut expectimax = Expectimax::new();
    let mut board = GameEngine::insert_random_tile(0);
    board = GameEngine::insert_random_tile(board);
    println!("{}", GameEngine::to_str(board));
    let mut move_count = 0;
    while !GameEngine::is_game_over(board) {
        let direction = expectimax.get_next_move(board);
        if direction.is_none() {
            break;
        }
        move_count += 1;
        board = GameEngine::make_move(board, direction.unwrap());
        println!("{}", GameEngine::to_str(board));
    }
    println!("Moves made: {}, States considered: {}, Max states considered for a move: {}", move_count, expectimax.0, expectimax.1)
}
