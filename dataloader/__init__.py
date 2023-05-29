from .bert_loader import BERT_Loader

max_seq_lengths = {
                        'stackoverflow': 45,
                        'banking': 55,
                        'oos': 30,
                        'snips': 35,
                        'clinc': 45,
                        'atis': 70
                    }


backbone_loader_map = {
                            'bert': BERT_Loader,
                            'bert_doc': BERT_Loader,
                            'bert_norm': BERT_Loader,
                            'bert_seg': BERT_Loader,
                            'bert_disaware': BERT_Loader,
                            'bert_adv': BERT_Loader
                        }

benchmark_labels = {
    'oos':
            ['accept_reservations', 'account_blocked', 'alarm', 'application_status', 'apr',
             'are_you_a_bot', 'balance', 'bill_balance', 'bill_due', 'book_flight',
             'book_hotel', 'calculator', 'calendar', 'calendar_update', 'calories',
             'cancel', 'cancel_reservation', 'car_rental', 'card_declined', 'carry_on',
             'change_accent', 'change_ai_name', 'change_language', 'change_speed', 'change_user_name',
             'change_volume', 'confirm_reservation', 'cook_time', 'credit_limit', 'credit_limit_change',
             'credit_score', 'current_location', 'damaged_card', 'date', 'definition',
             'direct_deposit', 'directions', 'distance', 'do_you_have_pets', 'exchange_rate',
             'expiration_date', 'find_phone', 'flight_status', 'flip_coin', 'food_last',
             'freeze_account', 'fun_fact', 'gas', 'gas_type', 'goodbye',
             'greeting', 'how_busy', 'how_old_are_you', 'improve_credit_score', 'income',
             'ingredient_substitution', 'ingredients_list', 'insurance', 'insurance_change', 'interest_rate',
             'international_fees', 'international_visa', 'jump_start', 'last_maintenance', 'lost_luggage', 'make_call',
             'maybe', 'meal_suggestion', 'meaning_of_life', 'measurement_conversion', 'meeting_schedule',
             'min_payment', 'mpg', 'new_card', 'next_holiday', 'next_song',
             'no', 'nutrition_info', 'oil_change_how', 'oil_change_when', 'order',
             'order_checks', 'order_status', 'pay_bill', 'payday', 'pin_change',
             'play_music', 'plug_type', 'pto_balance', 'pto_request', 'pto_request_status',
             'pto_used', 'recipe', 'redeem_rewards', 'reminder', 'reminder_update',
             'repeat', 'replacement_card_duration', 'report_fraud', 'report_lost_card', 'reset_settings',
             'restaurant_reservation', 'restaurant_reviews', 'restaurant_suggestion', 'rewards_balance', 'roll_dice',
             'rollover_401k', 'routing', 'schedule_maintenance', 'schedule_meeting', 'share_location',
             'shopping_list', 'shopping_list_update', 'smart_home', 'spelling', 'spending_history',
             'sync_device', 'taxes', 'tell_joke', 'text', 'thank_you',
             'time', 'timer', 'timezone', 'tire_change', 'tire_pressure',
             'todo_list', 'todo_list_update', 'traffic', "transactions", "transfer",
             "translate", "travel_alert", "travel_notification", "travel_suggestion", "uber",
             "update_playlist", "user_name", "vaccines", "w2", "weather",
             "what_are_your_hobbies", "what_can_i_ask_you", "what_is_your_name", "what_song", "where_are_you_from",
             "whisper_mode", "who_do_you_work_for", "who_made_you", "yes"],

    'banking':
                [
                    "Refund_not_showing_up", "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support",
                    "automatic_top_up", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit", "beneficiary_not_allowed", "cancel_transfer",
                    "card_about_to_expire", "card_acceptance", "card_arrival", "card_delivery_estimate", "card_linking",
                    "card_not_working", "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate", "card_swallowed",
                    "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin", "compromised_card", "contactless_not_working",
                    "country_support", "declined_card_payment", "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
                    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate", "exchange_via_app",
                    "extra_charge_on_statement", "failed_transfer", "fiat_currency_support", "get_disposable_virtual_card", "get_physical_card",
                    "getting_spare_card", "getting_virtual_card", "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card",
                    "passcode_forgotten", "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
                    "pin_blocked", "receiving_money", "request_refund", "reverted_card_payment?", "supported_cards_and_currencies",
                    "terminate_account", "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque", "top_up_failed",
                    "top_up_limits", "top_up_reverted", "topping_up_by_card", "transaction_charged_twice", "transfer_fee_charged",
                    "transfer_into_account", "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity", "verify_my_identity",
                    "verify_source_of_funds", "verify_top_up", "virtual_card_not_working", "visa_or_mastercard", "why_verify_identity",
                    "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal"
                ],

    'stackoverflow':
                        [
                            "ajax", "apache", "bash", "cocoa", "drupal",
                            "excel", "haskell", "hibernate", "linq", "magento",
                            "matlab", "oracle", "osx", "qt", "scala",
                            "sharepoint", "spring", "svn", "visual-studio", "wordpress"
                        ],
        'snips':
                [
                    "AddToPlaylist", "BookRestaurant", "GetWeather", "PlayMusic", "RateBook",
                    "SearchCreativeWork", "SearchScreeningEvent"
                ],
        'clinc': ['yes', 'insurance_change', 'calendar_update',
           'replacement_card_duration', 'no', 'international_visa',
           'application_status', 'cook_time', 'distance', 'uber',
           'interest_rate', 'make_call', 'rollover_401k', 'todo_list_update',
           'international_fees', 'who_do_you_work_for', 'payday',
           'order_status', 'jump_start', 'pto_request', 'rewards_balance',
           'shopping_list_update', 'change_accent', 'current_location',
           'timezone', 'book_flight', 'whisper_mode', 'calories', 'todo_list',
           'what_is_your_name', 'maybe', 'credit_limit', 'play_music',
           'meal_suggestion', 'pto_request_status', 'roll_dice',
           'tire_change', 'definition', 'account_blocked', 'change_speed',
           'ingredients_list', 'flight_status', 'goodbye', 'update_playlist',
           'restaurant_suggestion', 'transactions', 'credit_limit_change',
           'greeting', 'flip_coin', 'travel_suggestion', 'book_hotel',
           'oil_change_when', 'what_are_your_hobbies', 'cancel_reservation',
           'date', 'insurance', 'confirm_reservation', 'calculator',
           'do_you_have_pets', 'bill_balance', 'change_volume',
           'oil_change_how', 'thank_you', 'expiration_date', 'food_last',
           'freeze_account', 'what_song', 'car_rental', 'pay_bill',
           'directions', 'balance', 'lost_luggage', 'schedule_maintenance',
           'last_maintenance', 'gas', 'vaccines', 'reminder', 'pto_balance',
           'calendar', 'are_you_a_bot', 'travel_alert', 'find_phone',
           'pin_change', 'next_song', 'timer', 'reset_settings', 'carry_on',
           'mpg', 'meeting_schedule', 'smart_home', 'credit_score', 'text',
           'next_holiday', 'card_declined', 'measurement_conversion',
           'user_name', 'exchange_rate', 'gas_type', 'spelling', 'how_busy',
           'how_old_are_you', 'sync_device', 'tell_joke', 'routing',
           'traffic', 'translate', 'improve_credit_score', 'direct_deposit',
           'accept_reservations', 'schedule_meeting', 'new_card',
           'what_can_i_ask_you', 'change_language', 'w2',
           'ingredient_substitution', 'weather', 'restaurant_reservation',
           'cancel', 'spending_history', 'repeat', 'pto_used', 'taxes',
           'report_fraud', 'order', 'redeem_rewards', 'recipe',
           'reminder_update', 'tire_pressure', 'plug_type', 'share_location',
           'nutrition_info', 'report_lost_card', 'income', 'meaning_of_life',
           'restaurant_reviews', 'order_checks', 'transfer', 'damaged_card',
           'time', 'travel_notification', 'change_ai_name', 'shopping_list',
           'who_made_you', 'fun_fact', 'change_user_name',
           'where_are_you_from', 'apr', 'bill_due', 'min_payment', 'alarm'],
    'atis': ['atis_flight', 'atis_flight_time', 'atis_airfare', 'atis_aircraft',
       'atis_ground_service', 'atis_airport', 'atis_airline',
       'atis_distance', 'atis_abbreviation', 'atis_ground_fare',
       'atis_quantity', 'atis_city', 'atis_flight_no', 'atis_capacity',
       'atis_meal', 'atis_restriction', 'atis_cheapest']
}