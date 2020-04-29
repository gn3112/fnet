import wget
import os

root = "http://rll.berkeley.edu/bigbird/aliases/931f836a4b/"

objs = ['export/3m_high_tack_spray_adhesive/rgbd.tgz', 'export/advil_liqui_gels/rgbd.tgz', 'export/aunt_jemima_original_syrup/rgbd.tgz', 'export/bai5_sumatra_dragonfruit/rgbd.tgz', 'export/band_aid_clear_strips/rgbd.tgz', 'export/band_aid_sheer_strips/rgbd.tgz', 'export/blue_clover_baby_toy/rgbd.tgz', 'export/bumblebee_albacore/rgbd.tgz', 'export/campbells_chicken_noodle_soup/rgbd.tgz', 'export/campbells_soup_at_hand_creamy_tomato/rgbd.tgz', 'export/canon_ack_e10_box/rgbd.tgz', 'export/cheez_it_white_cheddar/rgbd.tgz', 'export/chewy_dipps_chocolate_chip/rgbd.tgz', 'export/chewy_dipps_peanut_butter/rgbd.tgz', 'export/cholula_chipotle_hot_sauce/rgbd.tgz', 'export/cinnamon_toast_crunch/rgbd.tgz', 'export/clif_crunch_chocolate_chip/rgbd.tgz', 'export/clif_crunch_peanut_butter/rgbd.tgz', 'export/clif_crunch_white_chocolate_macademia_nut/rgbd.tgz', 'export/clif_z_bar_chocolate_chip/rgbd.tgz', 'export/clif_zbar_chocolate_brownie/rgbd.tgz', 'export/coca_cola_glass_bottle/rgbd.tgz', 'export/coffee_mate_french_vanilla/rgbd.tgz', 'export/colgate_cool_mint/rgbd.tgz', 'export/crayola_24_crayons/rgbd.tgz', 'export/crayola_yellow_green/rgbd.tgz', 'export/crest_complete_minty_fresh/rgbd.tgz', 'export/crystal_hot_sauce/rgbd.tgz', 'export/cup_noodles_chicken/rgbd.tgz', 'export/cup_noodles_shrimp_picante/rgbd.tgz', 'export/detergent/rgbd.tgz', 'export/dove_beauty_cream_bar/rgbd.tgz', 'export/dove_go_fresh_burst/rgbd.tgz', 'export/dove_pink/rgbd.tgz', 'export/eating_right_for_healthy_living_apple/rgbd.tgz', 'export/eating_right_for_healthy_living_blueberry/rgbd.tgz', 'export/eating_right_for_healthy_living_mixed_berry/rgbd.tgz', 'export/eating_right_for_healthy_living_raspberry/rgbd.tgz', 'export/expo_marker_red/rgbd.tgz', 'export/fruit_by_the_foot/rgbd.tgz', 'export/gushers_tropical_flavors/rgbd.tgz', 'export/haagen_dazs_butter_pecan/rgbd.tgz', 'export/haagen_dazs_cookie_dough/rgbd.tgz', 'export/hersheys_bar/rgbd.tgz', 'export/hersheys_cocoa/rgbd.tgz', 'export/honey_bunches_of_oats_honey_roasted/rgbd.tgz', 'export/honey_bunches_of_oats_with_almonds/rgbd.tgz', 'export/hunts_paste/rgbd.tgz', 'export/hunts_sauce/rgbd.tgz', 'export/ikea_table_leg_blue/rgbd.tgz', 'export/krylon_crystal_clear/rgbd.tgz', 'export/krylon_low_odor_clear_finish/rgbd.tgz', 'export/krylon_matte_finish/rgbd.tgz', 'export/krylon_short_cuts/rgbd.tgz', 'export/listerine_green/rgbd.tgz', 'export/mahatma_rice/rgbd.tgz', 'export/mom_to_mom_butternut_squash_pear/rgbd.tgz', 'export/mom_to_mom_sweet_potato_corn_apple/rgbd.tgz', 'export/motts_original_assorted_fruit/rgbd.tgz', 'export/nature_valley_crunchy_oats_n_honey/rgbd.tgz', 'export/nature_valley_crunchy_variety_pack/rgbd.tgz', 'export/nature_valley_gluten_free_roasted_nut_crunch_almond_crunch/rgbd.tgz', 'export/nature_valley_granola_thins_dark_chocolate/rgbd.tgz', 'export/nature_valley_soft_baked_oatmeal_squares_cinnamon_brown_sugar/rgbd.tgz', 'export/nature_valley_soft_baked_oatmeal_squares_peanut_butter/rgbd.tgz', 'export/nature_valley_sweet_and_salty_nut_almond/rgbd.tgz', 'export/nature_valley_sweet_and_salty_nut_cashew/rgbd.tgz', 'export/nature_valley_sweet_and_salty_nut_peanut/rgbd.tgz', 'export/nature_valley_sweet_and_salty_nut_roasted_mix_nut/rgbd.tgz', 'export/nice_honey_roasted_almonds/rgbd.tgz', 'export/nutrigrain_apple_cinnamon/rgbd.tgz', 'export/nutrigrain_blueberry/rgbd.tgz', 'export/nutrigrain_cherry/rgbd.tgz', 'export/nutrigrain_chotolatey_crunch/rgbd.tgz', 'export/nutrigrain_fruit_crunch_apple_cobbler/rgbd.tgz', 'export/nutrigrain_fruit_crunch_strawberry_parfait/rgbd.tgz', 'export/nutrigrain_harvest_blueberry_bliss/rgbd.tgz', 'export/nutrigrain_harvest_country_strawberry/rgbd.tgz', 'export/nutrigrain_raspberry/rgbd.tgz', 'export/nutrigrain_strawberry/rgbd.tgz', 'export/nutrigrain_strawberry_greek_yogurt/rgbd.tgz', 'export/nutrigrain_toffee_crunch_chocolatey_toffee/rgbd.tgz', 'export/palmolive_green/rgbd.tgz', 'export/palmolive_orange/rgbd.tgz', 'export/paper_cup_holder/rgbd.tgz', 'export/paper_plate/rgbd.tgz', 'export/pepto_bismol/rgbd.tgz', 'export/pop_secret_butter/rgbd.tgz', 'export/pop_secret_light_butter/rgbd.tgz', 'export/pop_tarts_strawberry/rgbd.tgz', 'export/pringles_bbq/rgbd.tgz', 'export/progresso_new_england_clam_chowder/rgbd.tgz', 'export/quaker_big_chewy_chocolate_chip/rgbd.tgz', 'export/quaker_big_chewy_peanut_butter_chocolate_chip/rgbd.tgz', 'export/quaker_chewy_chocolate_chip/rgbd.tgz', 'export/quaker_chewy_dipps_peanut_butter_chocolate/rgbd.tgz', 'export/quaker_chewy_low_fat_chocolate_chunk/rgbd.tgz', 'export/quaker_chewy_peanut_butter/rgbd.tgz', 'export/quaker_chewy_peanut_butter_chocolate_chip/rgbd.tgz', 'export/quaker_chewy_smores/rgbd.tgz', 'export/red_bull/rgbd.tgz', 'export/red_cup/rgbd.tgz', 'export/ritz_crackers/rgbd.tgz', 'export/softsoap_clear/rgbd.tgz', 'export/softsoap_gold/rgbd.tgz', 'export/softsoap_green/rgbd.tgz', 'export/softsoap_purple/rgbd.tgz', 'export/softsoap_white/rgbd.tgz', 'export/south_beach_good_to_go_dark_chocolate/rgbd.tgz', 'export/south_beach_good_to_go_peanut_butter/rgbd.tgz', 'export/spam/rgbd.tgz', 'export/spongebob_squarepants_fruit_snaks/rgbd.tgz', 'export/suave_sweet_guava_nectar_body_wash/rgbd.tgz', 'export/sunkist_fruit_snacks_mixed_fruit/rgbd.tgz', 'export/tapatio_hot_sauce/rgbd.tgz', 'export/v8_fusion_peach_mango/rgbd.tgz', 'export/v8_fusion_strawberry_banana/rgbd.tgz', 'export/vo5_extra_body_volumizing_shampoo/rgbd.tgz', 'export/vo5_split_ends_anti_breakage_shampoo/rgbd.tgz', 'export/vo5_tea_therapy_healthful_green_tea_smoothing_shampoo/rgbd.tgz', 'export/white_rain_sensations_apple_blossom_hydrating_body_wash/rgbd.tgz', 'export/white_rain_sensations_ocean_mist_hydrating_body_wash/rgbd.tgz', 'export/white_rain_sensations_ocean_mist_hydrating_conditioner/rgbd.tgz', 'export/windex/rgbd.tgz', 'export/zilla_night_black_heat/rgbd.tgz']

for obj in objs:
    path = root + objs
    wget.download(path, "/home/georgesnomicos/bigbird/")